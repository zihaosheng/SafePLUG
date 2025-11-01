
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="EXP_NAME"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,2 --master_port=65001 train_safeplug.py \
  --version="./hf/lora-text" \
  --vision_tower='openai/clip-vit-large-patch14' \
  --image_tower='LanguageBind/LanguageBind_Image' \
  --video_tower='LanguageBind/LanguageBind_Video_merge' \
  --data_path './data/mini/grounding-mini.json' \
  --val_data_path './data/mini/grounding-mini.json' \
  --image_folder='/data' \
  --video_folder='/data' \
  --vision_pretrained="/data/huggingface-models/sam_vit_h_4b8939.pth" \
  --exp_name=$exp_name \
  --epochs=20 \
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
  --iou_loss_weight 0 \
  --focal_loss_weight 1.0 \
  --lora_r 8 \
  --lora_target_modules "gate_proj,up_proj,down_proj" \
  --sft_modules "mask_decoder,text_hidden_fcs" \
  --lr 0.0001 \
  --save_steps 300 \
  --sam_img_size 1024 \
  --train_mask_decoder \
  --no_eval \
  2>&1|tee -a runs/$exp_name/$time.log
