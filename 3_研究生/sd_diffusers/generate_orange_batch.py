from pathlib import Path
import argparse

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import trange


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUT_DIR = Path("/mnt/e/mastercode/data/orange_wuxi_sd_generated")

PROMPTS = [
    "photorealistic citrus orchard, green citrus fruits on branches, dense leaves, natural sunlight, leaf occlusion, close-up farm image",
    "realistic orange orchard, clusters of immature green citrus, overlapping fruits, severe leaf and branch occlusion, complex canopy, handheld RGB image",
    "green citrus fruits on tree, visible fruit stem area, dense orchard canopy, fruits suitable for bagging, natural farm environment",
    "close-up realistic citrus tree, many green fruits growing densely, leaves and branches occluding fruits, natural outdoor shadows",
]

NEGATIVE_PROMPT = (
    "cartoon, painting, fake fruit, deformed fruit, text, watermark, logo, "
    "human hand, basket, harvested fruit, sliced fruit, blurry, overexposed"
)

DEFAULT_NUM_IMAGES = 200
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 6.0
DEFAULT_BASE_SEED = 20260708


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-generate citrus orchard images with diffusers.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--variant", default="fp16", help="Weight variant. Use fp16 for 4GB CUDA GPUs, or none.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--num", type=int, default=DEFAULT_NUM_IMAGES, help="Number of images to generate.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Generated image width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Generated image height.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="Base random seed.")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use model CPU offload when CUDA memory is tight. Slower, but safer for 4GB GPUs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    variant = None if args.variant.lower() in {"", "none", "null"} else args.variant
    pretrained_kwargs = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if variant:
        pretrained_kwargs["variant"] = variant

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        **pretrained_kwargs,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        if args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    else:
        pipe.to("cpu")

    generator_device = "cpu" if args.cpu_offload or not torch.cuda.is_available() else "cuda"

    for idx in trange(args.num, desc="Generating"):
        prompt = PROMPTS[idx % len(PROMPTS)]
        generator = torch.Generator(device=generator_device).manual_seed(args.seed + idx)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]

        image.save(args.out / f"sd_orange_{idx:04d}.jpg", quality=95)

    print(f"Saved {args.num} images to {args.out}")


if __name__ == "__main__":
    main()
