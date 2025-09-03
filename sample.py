import os
import json
import argparse
import logging

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan import WanT2V
from wan.distributed.util import init_distributed_group

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, info_list):
        self.info_list = info_list

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        return self.info_list[idx]


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    init_distributed_group()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, local_rank, world_size


def init_wan_pipeline(args, rank, local_rank, world_size):
    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    elif isinstance(args.sample_guide_scale, float):
        args.sample_guide_scale = (args.sample_guide_scale, args.sample_guide_scale)
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    pipeline = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp if world_size > 1 else False,
        dit_fsdp=args.dit_fsdp if world_size > 1 else False,
        use_sp=(args.ulysses_size > 1) if world_size > 1 else False,
        t5_cpu=args.t5_cpu,
    )
    return pipeline


def generate_samples(args):
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        logger.info("Loading prompts...")
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            info_list = json.load(f)
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        info_list = None

    if world_size > 1:
        obj_list = [info_list]
        dist.broadcast_object_list(obj_list, src=0)
        info_list = obj_list[0]

    dataset = PromptDataset(info_list)
    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        collate_fn=lambda x: x[0],
    )

    pipeline = init_wan_pipeline(args, rank, local_rank, world_size)

    for item in tqdm(dataloader, disable=rank != 0):
        prompt = item["prompt"]
        safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]

        for sample_idx in range(args.num_samples):
            output_filename = f"{safe_prompt}-{sample_idx}.mp4"
            output_path = os.path.join(args.output_dir, output_filename)
            if os.path.exists(output_path):
                logger.info(f"Skipping existing file: {output_path}")
                continue
            try:
                video = pipeline.generate(
                    prompt,
                    size=SIZE_CONFIGS[args.size],
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=(args.base_seed + sample_idx) if args.base_seed >= 0 else None,
                    offload_model=args.offload_model,
                )
                if rank == 0 and video is not None:
                    from wan.utils.utils import save_video

                    save_video(
                        tensor=video[None],
                        save_file=output_path,
                        fps=pipeline.config.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    logger.info(f"Saved video to {output_path}")
            except Exception as e:
                logger.error(f"Error generating {safe_prompt}-{sample_idx}: {e}")
                continue


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 T2V sampling script")
    parser.add_argument("--task", type=str, default="t2v-A14B", help="Task type")
    parser.add_argument("--size", type=str, default="1280*720", help="Output size")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to model checkpoints")
    parser.add_argument("--prompt_file", type=str, required=True, help="JSON file with prompts")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--num_samples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale")
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampling solver")
    parser.add_argument("--frame_num", type=int, default=None, help="Number of frames")
    parser.add_argument("--base_seed", type=int, default=1, help="Base random seed")

    parser.add_argument("--t5_fsdp", action="store_true", help="Use FSDP for T5")
    parser.add_argument("--dit_fsdp", action="store_true", help="Use FSDP for DiT")
    parser.add_argument("--t5_cpu", action="store_true", help="Place T5 on CPU")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism size")
    parser.add_argument("--offload_model", type=bool, default=None, help="Offload model to CPU after forward")

    return parser.parse_args()


def main():
    args = parse_args()
    generate_samples(args)


if __name__ == "__main__":
    main()
