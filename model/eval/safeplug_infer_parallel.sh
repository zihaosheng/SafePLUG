#!/bin/bash

CUDA_VISIBLE_DEVICES="0,2"
gpu_list="0,2,2"
echo GPU_LIST: $gpu_list
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo CHUNKS: $CHUNKS

# CKPT="safeplug-7b-stage2-video"
ROOT_PATH="/home/sky-lab/SHENG_code/SafePLUG"
DATA_TYPE="test"
answer_type='open'
PORT=64500

# multiple SPLITs can be configured here
for SPLIT in DoTA_caption_test DoTA_temporal_grounding_test MM_AU_caption_test MM_AU_temporal_grounding_test MM_AU_region_test; do
# for SPLIT in MM_AU_region_test; do
    echo ">>> Processing SPLIT: $SPLIT"
    
    # set different CKPTs for specific SPLITs
    # if [ "$SPLIT" == "MM_AU_region_test" ]; then
    #     CKPT="safeplug-7b-stage2-v2"
    # else
    #     CKPT="safeplug-7b-stage3-v2"
    # fi
    CKPT="safeplug-7b-stage4-v2-3"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        PORT=$((PORT-1))
        echo "Launching chunk $IDX on GPU ${GPULIST[$IDX]} for $SPLIT"
        deepspeed --include=localhost:${GPULIST[$IDX]} --master_port=$PORT model/eval/safeplug_vqa_infer.py \
            --version="$ROOT_PATH/hf/$CKPT" \
            --vision_tower='openai/clip-vit-large-patch14' \
            --answer_type=$answer_type \
            --image_folder='/data' \
            --video_folder '/data' \
            --vision_pretrained "/data/huggingface-models/sam_vit_h_4b8939.pth" \
            --val_data_path ./data/test/$SPLIT.json \
            --answers_file $ROOT_PATH/baseline_eval/$CKPT/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl \
            --eval_vqa \
            --region_fea_adapter \
            --moe_enable \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    output_file=$ROOT_PATH/baseline_eval/$CKPT/$DATA_TYPE/$SPLIT.jsonl
    > "$output_file"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_PATH/baseline_eval/$CKPT/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    echo "<<< Finished SPLIT: $SPLIT"
done
