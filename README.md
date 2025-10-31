<div id="top" align="center">
<p align="center">
  <strong>
    <h2 align="center">SafePLUG: Empowering Multimodal LLMs with Pixel-Level Insight and Temporal Grounding for Traffic Accident Understanding</h2>
    <h3 align="center"><a href="https://zihaosheng.github.io/SafePLUG/">Website</a> | <a href="https://arxiv.org/abs/2508.06763">Paper</a> | <a href="https://huggingface.co/zihaosheng">Hugging Face</a>  </h3>
  </strong>
</p>
</div>

<br/>


## Highlights
- SafePLUG Framework - Developed a novel multimodal large language model framework that integrates pixel-level understanding and temporal grounding for fine-grained traffic accident analysis.
- Comprehensive Accident Dataset - Built a large-scale benchmark with region QA, pixel-level grounding, accident description, and temporal localization tasks, featuring detailed bounding boxes, segmentation masks, and event boundaries.
- State-of-the-Art Performance - Outperforms strong baselines on four key tasks and generalizes well to different shapes and positions of visual prompts.


## 🛠️ Installation
```
conda create -n safeplug python=3.10 -y
conda activate safeplug
pip install -r requirements.txt
pip install flash-attn==2.5.2 --no-build-isolation
```

## 📀 Train

### Train LoRA Text Branch
```Shell
sh scripts/train_lora_text.sh
```
### Train LoRA Mask Branch
```Shell
sh scripts/train_lora_mask.sh
```

### Merge LoRA Weights and Save Full Model

When training is finished, to get the full model weight:
```bash
cd ./runs/safeplug-7b-stage2/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```bash
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --vision_pretrained="PATH_TO_SAM" \
  --save_path="PATH_TO_SAVED_MODEL"
```

For example:
```shell
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="/data/huggingface-models/llava-v1.5-7b" \
  --weight="./runs/safeplug-7b-stage2/pytorch_model.bin" \
  --save_path="./hf/safeplug-7b-stage2" \
  --vision_pretrained="/data/huggingface-models/sam_vit_h_4b8939.pth" \
  # --lora_r 8 \  # uncomment these lines if seg
  # --lora_target_modules "gate_proj,up_proj,down_proj" \
  # --sft_modules "mask_decoder,text_hidden_fcs"
```

<span style="color:red">**Note**</span>: When using `merge_lora_weights_and_save_hf_model.py`, remember to modify the parameters in the parser to ensure they are consistent with those used during training.


## Gradio Web UI Demo
1. launch the server controller
```shell
python -m model.serve.controller --host 0.0.0.0 --port 64000
```
2. launch the web server
```shell
python -m model.serve.gradio_web_server --controller http://localhost:64000 --model-list-mode reload --add_region_feature --port 64001 
```
3. launch the model worker
```shell
torchrun --nproc_per_node=1 -m model.serve.model_worker --host localhost --controller http://localhost:64000 --port 64002 --worker http://localhost:64002 --model-path "/home/sky-lab/SHENG_code/SafePLUG/hf/safeplug-7b-stage2" --add_region_feature --device_map cuda --vision_pretrained /data/huggingface-models/sam_vit_h_4b8939.pth
```
Note: You must use torchrun, otherwise loading MPI will cause it to freeze.

## Evaluation
### Pixel Grounding
```Shell
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:1 --master_port=64995 model/eval/safeplug_vqa_infer.py \
    --version="/home/sky-lab/SHENG_code/SafePLUG/hf/safeplug-7b-stage2-Grounding-DoTA" \
    --vision_tower='openai/clip-vit-large-patch14' \
    --answer_type='open' \
    --val_data_path='/home/sky-lab/SHENG_code/SafePLUG/data/test/DoTA_grounding_test.json' \
    --image_folder='/data' \
    --vision_pretrained="/data/huggingface-models/sam_vit_h_4b8939.pth" \
    --eval_seg \
    --region_fea_adapter \
    # --vis_mask  # uncomment this line to save mask
```


### Region_VQA & VQA
Infer to generate the prediction jsonl file.
```Shell
bash model/eval/safeplug_infer_parallel.sh
```

Calcuate the metrics.

```Shell
python model/eval/cal_metric.py \
    --pred="/path/to/the/jsonl_file"
```

Use GPT eval
```shell
OPENAI_API_KEY="sk-***" python model/eval/gpt_eval.py \
    --answer ./baseline_eval/safeplug-7b-stage2-video/infer_res/test/DoTA_caption_test.jsonl
```

### Temporal Grounding
```shell
python model/eval/cal_temporal_grounding.py \
    --pred="/path/to/the/jsonl_file" \
```

### Baseline Segmentation
```shell
python model/eval/cal_baseline_seg_metrics.py  \
    --p=./baseline_eval/Qwen2.5-VL-72B-Instruct/DoTA_scene_grounding_test.jsonl 
```
