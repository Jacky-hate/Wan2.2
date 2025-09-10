# generate_t2v.py
# Copyright 2024-2025 The Alibaba Wan Team Authors.

import argparse, logging, os, random, sys, warnings, torch, torch.distributed as dist
from datetime import datetime
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video, str2bool


warnings.filterwarnings("ignore")

EXAMPLE_PROMPT = {
    "t2v-A14B": {"prompt": "Two anthropomorphic cats ..."}
}

DEFAULT_CFG = WAN_CONFIGS["t2v-A14B"]

DEFAULT_GUIDE_SCALE = list(DEFAULT_CFG.sample_guide_scale)

def _validate_args(args):
    assert args.ckpt_dir, "Please specify --ckpt_dir"
    assert args.task in WAN_CONFIGS and args.task.startswith("t2v"), "Only T2V tasks supported"
    cfg = WAN_CONFIGS[args.task]
    if args.task != "t2v-A14B":
        if args.sample_steps == DEFAULT_CFG.sample_steps:
            args.sample_steps = cfg.sample_steps
        if args.frame_num == DEFAULT_CFG.frame_num:
            args.frame_num = cfg.frame_num
        if args.sample_shift == DEFAULT_CFG.sample_shift:
            args.sample_shift = cfg.sample_shift
        if args.sample_guide_scale == DEFAULT_GUIDE_SCALE:

            args.sample_guide_scale = list(cfg.sample_guide_scale)
    if isinstance(args.sample_guide_scale, list):
        if len(args.sample_guide_scale) == 1:
            args.sample_guide_scale = args.sample_guide_scale[0]
        elif len(args.sample_guide_scale) == 2:
            args.sample_guide_scale = tuple(args.sample_guide_scale)
        else:
            raise ValueError("--sample_guide_scale expects one or two floats")
            
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"{args.size} not supported, choose from {SUPPORTED_SIZES[args.task]}"
    )
    # prompt files
    assert os.path.isfile(args.prompt_file_input), "--prompt_file_input not found"
    assert os.path.isfile(args.prompt_file_name), "--prompt_file_name not found"

def _parse_args():
    p = argparse.ArgumentParser("Wan T2V batch sampler")
    p.add_argument("--task", default="t2v-A14B", choices=[k for k in WAN_CONFIGS if k.startswith("t2v")])
    p.add_argument("--size", default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--frame_num", type=int, default=DEFAULT_CFG.frame_num)
    p.add_argument("--ckpt_dir", default="/mnt/petrelfs/zoukai/model/Wan2.2-T2V-A14B")
    p.add_argument("--output_dir", default="/mnt/petrelfs/zoukai/data/Wan2.2-T2V-A14B")
    p.add_argument("--prompt_file_input", required=True, help="txt，每行一个生成 prompt")
    p.add_argument("--prompt_file_name", required=True, help="txt，每行一个命名 prompt")
    p.add_argument("--split", default="1/1")
    p.add_argument("--offload_model", type=str2bool, default=None)
    p.add_argument("--ulysses_size", type=int, default=1)
    p.add_argument("--ring_size", type=int, default=1)
    p.add_argument("--t5_fsdp", action="store_true")
    p.add_argument("--t5_cpu",  action="store_true")
    p.add_argument("--dit_fsdp", action="store_true")
    p.add_argument("--base_seed", type=int, default=-1)
    p.add_argument("--sample_solver", default="unipc", choices=["unipc","dpm++"])
    p.add_argument("--sample_steps", type=int, default=DEFAULT_CFG.sample_steps)
    p.add_argument("--sample_shift", type=float, default=DEFAULT_CFG.sample_shift)
    p.add_argument(
        "--sample_guide_scale",
        type=float,
        nargs="*",
        default=DEFAULT_GUIDE_SCALE,
        help="Classifier free guidance scale (one or two floats)",
    )

    p.add_argument("--reverse", action="store_true", help="是否反向生成")
    args = p.parse_args()
    _validate_args(args)
    return args

def _init_logging(rank):
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )

def load_prompt_files(input_path, name_path):
    with open(input_path, "r", encoding="utf-8") as f:
        gen_prompts = [l.strip() for l in f if l.strip()]
    with open(name_path, "r", encoding="utf-8") as f:
        name_prompts = [l.strip() for l in f if l.strip()]
    assert len(gen_prompts) == len(name_prompts), "两份 prompt 行数需一致"
    return list(zip(gen_prompts, name_prompts))

def generate(args):
    rank      = int(os.getenv("RANK", 0))
    world_sz  = int(os.getenv("WORLD_SIZE", 1))
    local_rank= int(os.getenv("LOCAL_RANK", 0))
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_sz > 1 else True
    if world_sz > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_sz)
        
    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_sz, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, 0); args.base_seed = base_seed[0]

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Config: {cfg}")

    from wan import WanT2V
    wan_t2v = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    prompt_pairs = load_prompt_files(args.prompt_file_input, args.prompt_file_name)
    os.makedirs(args.output_dir, exist_ok=True)

    sp_idx, sp_tot = map(int, args.split.split("/"))
    logging.info(f"Split {sp_idx}/{sp_tot} for {len(prompt_pairs)} prompts")
    if args.reverse:
        prompt_pairs = prompt_pairs[::-1]
    logging.info("----------sample args----------")
    logging.info(f"base_seed: {args.base_seed}")
    logging.info(f"sample_steps: {args.sample_steps}")
    logging.info(f"sample_shift: {args.sample_shift}")
    logging.info(f"sample_guide_scale: {args.sample_guide_scale}")
    logging.info(f"frame_num: {args.frame_num}")
    logging.info(f"offload_model: {args.offload_model}")
    logging.info(f"sample_solver: {args.sample_solver}")
    logging.info(f"size: {args.size}")
    logging.info("--------sample args end---------")
    
    for line_idx, (gen_prompt, name_prompt) in enumerate(prompt_pairs):
        if line_idx % sp_tot != sp_idx:
            continue
        logging.info(f"[{line_idx}] prompt: {gen_prompt}")

        for idx in range(5):
            outfile = f"{name_prompt}-{idx}.mp4"
            outpath = os.path.join(args.output_dir, outfile)
            if os.path.exists(outpath):
                logging.info(f"Exists: {outfile} – skip"); continue
            seed = args.base_seed + idx
            video = wan_t2v.generate(
                gen_prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=seed,
                offload_model=args.offload_model,
            )
            if rank == 0:
                save_video(
                    tensor=video[None],
                    save_file=outpath,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                logging.info(f"Saved to {outpath}")



"""Example usage:
srun -p video-aigc-3 --nodes=1 --gres=gpu:8 --cpus-per-task=16 \
 torchrun --nproc_per_node=8 --master_port=29525 generate_t2v.py \
 --task t2v-A14B \
 --size 1280*720 \
 --ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.2-T2V-A14B \
 --dit_fsdp --t5_fsdp --ulysses_size 8 \
 --output_dir /mnt/petrelfs/zoukai/data/Wan2.2-T2V-A14B \
 --prompt_file_input /path/to/prompts.txt \
 --prompt_file_name  /path/to/names.txt \
 --split 0/1
"""
if __name__ == "__main__":
    generate(_parse_args())
