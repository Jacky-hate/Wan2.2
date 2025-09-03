import os
import json
import argparse
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from datetime import datetime
from tqdm import tqdm
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan import WanI2V

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class I2VDataset(Dataset):
    def __init__(self, info_list, image_folder):
        self.info_list = info_list
        self.image_folder = image_folder
        
    def __len__(self):
        return len(self.info_list)
    
    def __getitem__(self, idx):
        info = self.info_list[idx]
        image_path = os.path.join(self.image_folder, info["image_name"])
        
        try:
            image = Image.open(image_path).convert("RGB")
            return {
                "image": image,
                "prompt_aug": info["prompt_aug"],
                "prompt_en": info["prompt_en"]
            }
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise


def setup_distributed():
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']
    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if world_size > 1:
        # Initialize the process group first
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        # Then set the CUDA device
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def init_wan_pipeline(args, rank, local_rank, world_size):
    
    
    cfg = WAN_CONFIGS[args.task]
    
    # Set default parameters if not provided
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50
    
    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
    
    if args.frame_num is None:
        args.frame_num = 81
    
    # Initialize pipeline with proper distributed settings
    pipeline = WanI2V(
        config=cfg,
        checkpoint_dir=args.model_path,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp if world_size > 1 else False,  # Disable FSDP if single GPU
        dit_fsdp=args.dit_fsdp if world_size > 1 else False,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1) if world_size > 1 else False,
        t5_cpu=args.t5_cpu,
    )
    
    return pipeline

def generate_samples(args):
    rank, local_rank, world_size = setup_distributed()
    
    # Only load and process data on rank 0
    if rank == 0:
        logger.info("Loading dataset info...")
        with open(args.i2v_path, 'r', encoding='utf-8') as f:
            info_list = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        info_list = None
    
    # Broadcast info_list to all ranks
    if world_size > 1:
        info_list = [None]
        dist.broadcast_object_list(info_list, src=0)
        info_list = info_list[0]
    
    # Create dataset and sampler
    dataset = I2VDataset(
        info_list=info_list,
        image_folder=args.image_folder,

    )
    
    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: x[0]  # 避免默认collate处理，直接返回单个dict
    )

    
    # Initialize pipeline
    pipeline = init_wan_pipeline(args, rank, local_rank, world_size)
    
    # Generate samples
    for batch in tqdm(dataloader, disable=rank != 0):
        image = batch["image"]
        prompt_aug = batch["prompt_aug"]
        prompt_en = batch["prompt_en"]

        # 清理prompt_en，用于安全的文件名
        safe_prompt = prompt_en

        for sample_idx in range(args.num_samples):
            output_filename = f"{safe_prompt}-{sample_idx}.mp4"
            output_path = os.path.join(args.output_dir, output_filename)

            # 检查视频是否已存在
            if os.path.exists(output_path):
                logger.info(f"Skipping existing file: {output_path}")
                continue

            try:
                # Generate video
                video = pipeline.generate(
                    prompt_aug,
                    image,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=(args.base_seed + sample_idx) if args.base_seed >= 0 else None,
                    offload_model=args.offload_model
                )

                # Save video (only on rank 0)
                if rank == 0:
                    from wan.utils.utils import cache_video
                    cache_video(
                        tensor=video[None],
                        save_file=output_path,
                        fps=pipeline.config.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1)
                    )

                    logger.info(f"Saved video to {output_path}")

            except Exception as e:
                logger.error(f"Error generating {safe_prompt}-{sample_idx}: {str(e)}")
                continue


def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale Wan I2V sampling")
    
    # Required arguments
    parser.add_argument("--task", type=str, default="i2v-14B", help="Task type")
    parser.add_argument("--size", type=str, default="1280*720", help="Output size")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--i2v_path", type=str, required=True, help="Path to I2V info JSON")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing input images")
    
    # Sampling parameters
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per prompt")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale")
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampling solver")
    parser.add_argument("--frame_num", type=int, default=None, help="Number of frames")
    parser.add_argument("--base_seed", type=int, default=1, help="Base seed for random generation")
    
    # Model configuration
    parser.add_argument("--t5_fsdp", action="store_true", help="Use FSDP for T5")
    parser.add_argument("--dit_fsdp", action="store_true", help="Use FSDP for DiT")
    parser.add_argument("--t5_cpu", action="store_true", help="Place T5 on CPU")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism size")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism size")
    parser.add_argument("--offload_model", type=bool, default=None, help="Offload model to CPU after forward")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables for distributed training
    os.environ["RANK"] = os.environ.get("RANK", "0")
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    
    generate_samples(args)

if __name__ == "__main__":
    main()