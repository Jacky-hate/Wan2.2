# generate_t2v.py
# Copyright 2024-2025 The Alibaba Wan Team Authors.

import argparse, logging, os, random, sys, warnings, torch, torch.distributed as dist
from datetime import datetime
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, str2bool

warnings.filterwarnings("ignore")

EXAMPLE_PROMPT = {
    "t2v-14B": {"prompt": "Two anthropomorphic cats ..."}
}

def _validate_args(args):
    assert args.ckpt_dir, "Please specify --ckpt_dir"
    assert args.task in WAN_CONFIGS and "t2v" in args.task, "Only T2V tasks supported"
    if args.sample_steps is None:
        args.sample_steps = 50
    if args.frame_num is None:
        args.frame_num = 81
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"{args.size} not supported, choose from {SUPPORTED_SIZES[args.task]}"
    )
    # prompt files
    assert os.path.isfile(args.prompt_file_input), "--prompt_file_input not found"
    assert os.path.isfile(args.prompt_file_name), "--prompt_file_name not found"

def _parse_args():
    p = argparse.ArgumentParser("Wan T2V batch sampler")
    p.add_argument("--task", default="t2v-14B", choices=[k for k in WAN_CONFIGS if "t2v" in k])
    p.add_argument("--size", default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--frame_num", type=int, default=None)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_dir", required=True)
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
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--sample_guide_scale", type=float, default=5.0)
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
                cache_video(
                    tensor=video[None],
                    save_file=outpath,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                logging.info(f"Saved to {outpath}")



"""
srun -p video-aigc-3 --nodes=1 --gres=gpu:8 --cpus-per-task=16 \
torchrun --nproc_per_node=8 --master_port=29525 generate_t2v.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.1-T2V-14B \
--dit_fsdp --t5_fsdp --ulysses_size 8 \
--output_dir /mnt/petrelfs/zoukai/data/Wan2.1-T2V-14B_aug_seed42 \
--prompt_file_input /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension_aug_wanx_seed42.txt \
--prompt_file_name  /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension.txt \
--sample_shift 5.0 \
--sample_guide_scale 5 \
--split 0/2 


srun -p video-aigc-3 --nodes=1 --gres=gpu:4 --cpus-per-task=16 \
torchrun --nproc_per_node=4 --master_port=29505 generate_t2v.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.1-T2V-1.3B \
--dit_fsdp --t5_fsdp --ulysses_size 4 \
--output_dir /mnt/petrelfs/zoukai/data/Wan2.1-T2V-1.3B_aug_seed42 \
--prompt_file_input /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension_aug_wanx_seed42.txt \
--prompt_file_name  /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension.txt \
--sample_shift 3.0 \
--sample_guide_scale 6 \
--split 1/2 

srun -p video-aigc-3 --nodes=1 --gres=gpu:4 --cpus-per-task=16 \
torchrun --nproc_per_node=4 --master_port=29502 generate_t2v.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.1-T2V-1.3B \
--dit_fsdp --t5_fsdp --ulysses_size 4 \
--output_dir /mnt/petrelfs/zoukai/data/Wan2.1-T2V-1.3B_aug_seed42 \
--prompt_file_input /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension_aug_wanx_seed42.txt \
--prompt_file_name  /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension.txt \
--sample_shift 3.0 \
--sample_guide_scale 6 \
--split 0/1 \
--reverse



srun -p video-aigc-3 --nodes=1 --gres=gpu:4 --cpus-per-task=16 \
torchrun --nproc_per_node=4 --master_port=29525 generate_t2v.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.1-T2V-1.3B \
--dit_fsdp --t5_fsdp --ulysses_size 4 \
--output_dir /mnt/petrelfs/zoukai/data/Wan2.1-T2V-1.3B_aug_new \
--prompt_file_input /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension_aug_wanx_new.txt \
--prompt_file_name  /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension.txt \
--sample_shift 3.0 \
--sample_guide_scale 6 \
--split 0/2 \
--reverse


srun -p video-aigc-3 --nodes=1 --gres=gpu:4 --cpus-per-task=16 \
torchrun --nproc_per_node=4 --master_port=29503 generate_t2v.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.1-T2V-1.3B \
--dit_fsdp --t5_fsdp --ulysses_size 4 \
--output_dir /mnt/petrelfs/zoukai/data/Wan2.1-T2V-1.3B_aug_new \
--prompt_file_input /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension_aug_wanx_new.txt \
--prompt_file_name  /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension.txt \
--sample_shift 3.0 \
--sample_guide_scale 6 \
--split 1/2 \
--reverse




srun -p video-aigc-3 --nodes=1 --gres=gpu:4 --cpus-per-task=16 \
torchrun --nproc_per_node=4 generate_t2v.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.1-T2V-1.3B \
--dit_fsdp --t5_fsdp --ulysses_size 4 \
--output_dir /mnt/petrelfs/zoukai/data/Wan2.1-T2V-1.3B_aug \
--prompt_file_input /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension_aug_wanx.txt \
--prompt_file_name  /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension.txt \
--split 0/2

srun -p video-aigc-3 --nodes=1 --gres=gpu:4 --cpus-per-task=16 \
torchrun --nproc_per_node=4 --master_port=29503 generate_t2v.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /mnt/petrelfs/zoukai/model/Wan2.1-T2V-1.3B \
--dit_fsdp --t5_fsdp --ulysses_size 4 \
--output_dir /mnt/petrelfs/zoukai/data/Wan2.1-T2V-1.3B_aug \
--prompt_file_input /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension_aug_wanx.txt \
--prompt_file_name  /mnt/petrelfs/zoukai/code/VBench/prompts/all_dimension.txt \
--split 1/2

"""
if __name__ == "__main__":
    generate(_parse_args())
