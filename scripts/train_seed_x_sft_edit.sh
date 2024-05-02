
PROJ_PATH='SEED-X'
exp_name='seed_x_sft_edit'
OUTPUT_PATH=SEED-X/train_output/${exp_name}

mkdir -p $OUTPUT_PATH

export PYTHONPATH=SEED-X/proj/peft/src:$PYTHONPATH

#torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=20008 --node_rank=$INDEX \
torchrun --nproc_per_node=8 \
    ${PROJ_PATH}/src/train/train_seed_x_sft.py \
    --image_transform ${PROJ_PATH}/configs/processer/qwen_448_transform.yaml \
    --tokenizer ${PROJ_PATH}/configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml \
    --visual_encoder ${PROJ_PATH}/configs/visual_encoder/qwen_vitg_448.yaml \
    --llm_model ${PROJ_PATH}/configs/clm_models/llm_seed_x_lora.yaml \
    --agent_model ${PROJ_PATH}/configs/clm_models/agent_seed_x.yaml \
    --train_dataset ${PROJ_PATH}/configs/data/sft_edit.yaml \
    --output_dir ${OUTPUT_PATH} \
    --expr_name  ${exp_name} \
    --learning_rate 1e-4 \
    --batch_size 50 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 2 \
    --mixed_precision bf16 \
    --num_train_epochs 10 \
    --max_steps 20000 \
    --save_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_3.yaml \
    #--deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2_offload.yaml \


echo '--------------------------'
echo main training task done
echo '--------------------------'
