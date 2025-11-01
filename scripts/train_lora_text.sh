
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="EXP_NAME"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,2 --master_port=65001 train_safeplug.py \
  --version="/data/huggingface-models/llava-v1.5-7b" \
  --vision_tower='openai/clip-vit-large-patch14' \
  --pretrain_mm_mlp_adapter='/data/huggingface-models/Video-LLaVA-Pretrain-7B/mm_projector.bin' \
  --image_tower='LanguageBind/LanguageBind_Image' \
  --video_tower='LanguageBind/LanguageBind_Video_merge' \
  --data_path './data/mini/region-mini.json' \
  --val_data_path './data/mini/region-mini.json' \
  --image_folder='/data' \
  --video_folder='/data' \
  --vision_pretrained="/data/huggingface-models/sam_vit_h_4b8939.pth" \
  --exp_name=$exp_name \
  --epochs=6 \
  --batch_size=1 \
  --workers=8 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 2 \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 5.0 \
  --bce_loss_weight 1.0 \
  --lora_r 16 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --sft_modules "lm_head,embed_tokens,input_layernorm,post_attention_layernorm,mm_projector" \
  --lr 0.0001 \
  --no_eval \
  --save_steps 400 \
  2>&1|tee -a runs/$exp_name/$time.log
