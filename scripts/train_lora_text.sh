
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="safetygpt-7b-stage2-v1.1-resume"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=65001 train_safeplug.py \
  --version="/data/hugging-face-models/llava-v1.5-7b" \
  --vision_tower='openai/clip-vit-large-patch14' \
  --pretrain_mm_mlp_adapter='/data/hugging-face-models/Video-LLaVA-Pretrain-7B/mm_projector.bin' \
  --image_tower='LanguageBind/LanguageBind_Image' \
  --video_tower='LanguageBind/LanguageBind_Video_merge' \
  --data_path './data/mini/DoTA_caption_train.json||./data/mini/MM_AU_caption_train.json' \
  --val_data_path './data/mini/DoTA_caption_train.json' \
  --image_folder='/data/hugging-face-models' \
  --video_folder='/data/hugging-face-models' \
  --vision_pretrained="/data/hugging-face-models/sam_vit_h_4b8939.pth" \
  --exp_name=$exp_name \
  --epochs=6 \
  --batch_size=4 \
  --workers=8 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 8 \
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
