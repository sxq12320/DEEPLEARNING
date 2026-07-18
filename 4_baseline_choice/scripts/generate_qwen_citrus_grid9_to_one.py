"""Generate one citrus hard sample from multiple reference images."""

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
sys.modules["qwen_single_grid9"] = qwen_single
spec.loader.exec_module(qwen_single)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROMPT_TEMPLATE = """输入包含 {count} 张真实柑橘果园参考图片。请把这些图作为风格、场景、果实形态、叶片密度和遮挡关系的参考，生成一张新的完整图片，不要输出拼贴图、边框或分屏画面。

请生成一张单张完整的真实柑橘果园近景图片，用于未成熟柑橘幼果实例分割训练。

生成目标：比参考图更困难，但仍然真实自然。柑橘必须是绿色或青绿色未成熟幼果，果皮纹理、表面颗粒感、颜色分布和自然反光要接近参考图，不要变成熟橙色果实，不要变成商业摄影风格。

请综合这些参考图的视觉特点，适当增加未成熟柑橘幼果数量，尤其增加远处小目标、边缘小目标、叶片缝隙中的小目标和半遮挡小目标。果实应自然分布在枝叶之间，不要整齐排列。

请增加真实叶片和细枝条遮挡，遮挡应自然，部分呈细长条带状穿过果实表面，使果实可见区域出现凹陷边界、不完整轮廓和实例分离困难。不要完全遮住所有果实。

请增加相邻果实靠近、局部重叠、边界接近、果叶颜色相近、背景叶片复杂、局部阴影和斑驳光照。图像应明显比普通样本更难，但仍然清晰可人工重新标注实例分割 mask。

输出要求：单张完整图片，真实农业机器人果园拍摄风格，照片级真实感，无文字，无水印，无拼贴边框。"""

NEGATIVE_PROMPT = (
    qwen_single.NEGATIVE_PROMPT
    + ", collage, grid, nine panels, split screen, border, frame, contact sheet, photo montage"
)


def is_wan27_model(model: str) -> bool:
    return model.startswith("wan2.7")


def model_image_limit(model: str) -> int:
    if is_wan27_model(model):
        return 9
    if model.startswith("qwen-image"):
        return 6
    return 9


def iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_prompt(count: int) -> str:
    return PROMPT_TEMPLATE.format(count=count)


def build_payload(model: str, reference_images: list[Path], source_count: int, size: str) -> dict:
    parameters = {
        "n": 1,
        "size": size,
        "watermark": False,
    }
    if not is_wan27_model(model):
        parameters["prompt_extend"] = True
        parameters["negative_prompt"] = NEGATIVE_PROMPT

    content = [{"image": qwen_single.encode_image(path)} for path in reference_images]
    content.append({"text": build_prompt(source_count)})

    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": parameters,
    }


def generate_from_references(
    reference_images: list[Path],
    source_images: list[Path],
    output_dir: Path,
    api_csv: Path,
    model: str,
    size: str,
) -> tuple[Path, Path]:
    creds = qwen_single.read_api_csv(api_csv)
    endpoint = f"https://{creds['apiHost'].rstrip('/')}/api/v1/services/aigc/multimodal-generation/generation"
    payload = build_payload(model, reference_images, len(source_images), size)

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
    stem = f"refs{len(source_images)}_qwen_hard_{stamp}"
    image_path = output_dir / f"{stem}.png"
    meta_path = output_dir / f"{stem}.json"
    qwen_single.download_image(urls[0], image_path)

    metadata = {
        "source_images": [str(p) for p in source_images],
        "reference_images": [str(p) for p in reference_images],
        "output_image": str(image_path),
        "model": model,
        "size": size,
        "prompt": build_prompt(len(source_images)),
        "negative_prompt": NEGATIVE_PROMPT if not is_wan27_model(model) else "",
        "request_id": response_json.get("request_id"),
        "usage": response_json.get("usage"),
    }
    meta_path.write_text(qwen_single.json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(r"E:\mastercode\4_baseline_choice\ai_aug_multi"))
    parser.add_argument("--api-csv", type=Path, default=qwen_single.DEFAULT_API_CSV)
    parser.add_argument("--model", default="wan2.7-image-pro")
    parser.add_argument("--size", default="1152*2048")
    parser.add_argument("--group-size", type=int, default=9, help="How many source images are used for one generated image.")
    parser.add_argument("--limit-groups", type=int, default=1, help="0 means all groups.")
    parser.add_argument("--include-last", action="store_true", help="Also process the final incomplete group.")
    args = parser.parse_args()

    if args.group_size < 1:
        raise ValueError("--group-size must be >= 1.")
    image_limit = model_image_limit(args.model)
    if args.group_size > image_limit:
        raise ValueError(
            f"Model {args.model} supports at most {image_limit} input image(s) per request. "
            f"Please set --group-size <= {image_limit} or choose another model."
        )

    images = iter_images(args.input_dir)
    groups = []
    for i in range(0, len(images), args.group_size):
        group = images[i : i + args.group_size]
        if len(group) == args.group_size or (args.include_last and group):
            groups.append(group)
    if args.limit_groups > 0:
        groups = groups[: args.limit_groups]
    if not groups:
        raise RuntimeError(f"No image group found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"refs{args.group_size}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with log_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "source_images", "output_image", "metadata", "status", "error"])
        writer.writeheader()
        for group_idx, group in enumerate(groups, start=1):
            print(f"[{group_idx}/{len(groups)}] generating from {len(group)} images")
            try:
                reference_images = group
                output_image, metadata = generate_from_references(
                    reference_images,
                    group,
                    args.output_dir,
                    args.api_csv,
                    args.model,
                    args.size,
                )
                writer.writerow(
                    {
                        "group": group_idx,
                        "source_images": " | ".join(str(p) for p in group),
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
                        "group": group_idx,
                        "source_images": " | ".join(str(p) for p in group),
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
