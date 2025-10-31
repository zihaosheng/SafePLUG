import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.LISA import LISAForCausalLM
from model.SafetyGPT import SafetyGPTForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, ADD_OTHERS_TOKENS


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="/data/huggingface-models/llava-v1.5-7b"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="/data/huggingface-models/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--sam_img_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--image_tower", default="LanguageBind/LanguageBind_Image", type=str
    )
    parser.add_argument(
        "--video_tower", default="LanguageBind/LanguageBind_Video_merge", type=str
    )
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )
    #====== keep the same as training =============
    parser.add_argument("--lora_r", default=16, type=int)  
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str)
    parser.add_argument("--sft_modules", default="lm_head,embed_tokens,input_layernorm,post_attention_layernorm,mm_projector", type=str)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./lisa_model", type=str, required=True)
    parser.add_argument("--region_fea_adapter", action="store_true", default=False)
    parser.add_argument("--max_sample_point", default=512, type=int)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    for token_name in ADD_OTHERS_TOKENS:
        tokenizer.add_tokens(token_name, special_tokens=True)
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]


    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "image_tower": args.image_tower,
        "video_tower": args.video_tower,
        "region_fea_adapter": args.region_fea_adapter,
        "max_sample_point": args.max_sample_point,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = SafetyGPTForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    image_tower = model.get_model().get_image_tower()
    video_tower = model.get_model().get_video_tower()
    image_tower.to(dtype=torch_dtype)
    video_tower.to(dtype=torch_dtype)
    model.get_model().initialize_lisa_modules(model.get_model().config, sam_img_size=args.sam_img_size)

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "image_tower",
                                "video_tower",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for n, p in model.named_parameters():
            p.requires_grad = False

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    sft_modules = args.sft_modules.split(",")
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in sft_modules
            ]
        ):
            p.requires_grad = True


    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu")


    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('unexpected_keys',unexpected_keys)
    print('missing_keys',missing_keys)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "image_tower" in k or "video_tower" in k:
            continue
        state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])