# # Single GPU
# python large_scale_sample.py \
#     --task i2v-14B \
#     --size 1280*720 \
#     --model_path /mnt/petrelfs/zoukai/model/Wan2.1-I2V-14B-720P \
#     --output_dir /mnt/petrelfs/zoukai/data/Wan2.1-I2V-14B-720P \
#     --i2v_path /mnt/petrelfs/zoukai/code/VBench/vbench2_beta_i2v/vbench2_i2v_full_info_aug.json \
#     --image_folder /mnt/petrelfs/zoukai/code/VBench/vbench2_beta_i2v/data/crop/16-9 \
#     --num_samples 5

# Multi-GPU (8 GPUs)
srun -p video-aigc-3 --nodes=1  --gres=gpu:1 --cpus-per-task=16 torchrun --master_port 29501 --nproc_per_node=2 sample.py \
    --task i2v-14B \
    --size 1280*720 \
    --model_path /mnt/petrelfs/zoukai/model/Wan2.1-I2V-14B-720P \
    --output_dir /mnt/petrelfs/zoukai/data/Wan2.1-I2V-14B-720P \
    --i2v_path /mnt/petrelfs/zoukai/code/Wan2.1/vbench2_i2v_full_info_aug.json \
    --image_folder /mnt/petrelfs/zoukai/code/VBench/vbench2_beta_i2v/data/crop/16-9 \
    --num_samples 5 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --offload_model True