import argparse
import time

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download a diffusers Stable Diffusion model with resume support.")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id.")
    parser.add_argument("--variant", default="fp16", help="Weight variant to download. Use none for all weights.")
    parser.add_argument("--workers", type=int, default=1, help="Download workers. Use 1 for unstable networks.")
    parser.add_argument("--retries", type=int, default=20, help="Retry count for unstable network downloads.")
    parser.add_argument("--sleep", type=int, default=5, help="Seconds to sleep between retries.")
    return parser.parse_args()


def main():
    args = parse_args()
    variant = None if args.variant.lower() in {"", "none", "null"} else args.variant
    download_kwargs = {
        "repo_id": args.model,
        "resume_download": True,
        "max_workers": args.workers,
    }
    if variant:
        download_kwargs["allow_patterns"] = [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "feature_extractor/*",
            "text_encoder/config.json",
            f"text_encoder/*{variant}*",
            "unet/config.json",
            f"unet/*{variant}*",
            "vae/config.json",
            f"vae/*{variant}*",
        ]
    else:
        download_kwargs["ignore_patterns"] = [
            "*.onnx",
            "*.msgpack",
            "*.h5",
            "*.ot",
            "*.pb",
            "flax/*",
            "tf/*",
        ]

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            print(f"Download attempt {attempt}/{args.retries}")
            path = snapshot_download(**download_kwargs)
            print(f"Downloaded model to: {path}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < args.retries:
                time.sleep(args.sleep)

    raise RuntimeError(f"Download failed after {args.retries} attempts") from last_error


if __name__ == "__main__":
    main()
