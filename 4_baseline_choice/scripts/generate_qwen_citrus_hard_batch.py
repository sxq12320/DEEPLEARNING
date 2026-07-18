"""Batch-generate citrus hard-sample images from a folder."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_SCRIPT = SCRIPT_DIR / "generate_qwen_citrus_hard_sample.py"
spec = importlib.util.spec_from_file_location("qwen_single", SINGLE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {SINGLE_SCRIPT}")
qwen_single = importlib.util.module_from_spec(spec)
sys.modules["qwen_single"] = qwen_single
spec.loader.exec_module(qwen_single)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def generate_one(image: Path, output_dir: Path, api_csv: Path, model: str, size: str, seed: int | None) -> tuple[Path, Path]:
    creds = qwen_single.read_api_csv(api_csv)
    endpoint = f"https://{creds['apiHost'].rstrip('/')}/api/v1/services/aigc/multimodal-generation/generation"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": qwen_single.encode_image(image)},
                        {"text": qwen_single.PROMPT},
                    ],
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": qwen_single.NEGATIVE_PROMPT,
            "prompt_extend": True,
            "size": size,
        },
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    response = qwen_single.request_session().post(
        endpoint,
        headers={
            "Authorization": f"Bearer {creds['apiKey']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=600,
    )
    response_json = response.json()
    if response.status_code != 200 or response_json.get("code"):
        message = response_json.get("message", response.text[:500])
        code = response_json.get("code", response.status_code)
        raise RuntimeError(f"{code}: {message}")

    urls = qwen_single.extract_image_urls(response_json)
    if not urls:
        raise RuntimeError("No image URL found in response.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{image.stem}_qwen_hard_{stamp}"
    image_path = output_dir / f"{stem}.png"
    meta_path = output_dir / f"{stem}.json"

    qwen_single.download_image(urls[0], image_path)
    metadata = {
        "source_image": str(image),
        "output_image": str(image_path),
        "model": model,
        "size": size,
        "prompt": qwen_single.PROMPT,
        "negative_prompt": qwen_single.NEGATIVE_PROMPT,
        "request_id": response_json.get("request_id"),
        "usage": response_json.get("usage"),
    }
    meta_path.write_text(qwen_single.json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing source images.")
    parser.add_argument("--output-dir", type=Path, default=Path(r"E:\mastercode\4_baseline_choice\ai_aug_batch"))
    parser.add_argument("--api-csv", type=Path, default=qwen_single.DEFAULT_API_CSV)
    parser.add_argument("--model", default="qwen-image-2.0")
    parser.add_argument("--size", default="1152*2048", help="First-version 9:16 2K portrait size.")
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N images. 0 means all.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    images = iter_images(args.input_dir, args.recursive)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Images: {len(images)}")
    print(f"Log: {log_path}")

    with log_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "source_image", "output_image", "metadata", "status", "error"])
        writer.writeheader()

        for index, image in enumerate(images, start=1):
            print(f"[{index}/{len(images)}] {image.name}")
            try:
                output_image, metadata = generate_one(image, args.output_dir, args.api_csv, args.model, args.size, args.seed)
                writer.writerow(
                    {
                        "index": index,
                        "source_image": str(image),
                        "output_image": str(output_image),
                        "metadata": str(metadata),
                        "status": "ok",
                        "error": "",
                    }
                )
                print(f"  saved: {output_image.name}")
            except Exception as exc:
                writer.writerow(
                    {
                        "index": index,
                        "source_image": str(image),
                        "output_image": "",
                        "metadata": "",
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"  failed: {exc}")
            f.flush()

    print(f"Done. Log saved: {log_path}")


if __name__ == "__main__":
    main()
