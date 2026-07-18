"""Generate one citrus hard-sample image with Alibaba Qwen Image Edit API."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_CSV = Path(r"C:\Users\33836\Desktop\默认业务空间-apiKey-6164434.csv")
DEFAULT_IMAGE = Path(r"E:\mastercode\data\orange_wuxi\img\IMG_20260706_095525.jpg")
DEFAULT_OUTPUT_DIR = Path(r"E:\mastercode\4_baseline_choice\ai_aug_preview")

PROMPT = """基于输入参考图进行“中等强度困难化增强”，输出一张更复杂但仍然真实自然的柑橘果园图像，用于未成熟柑橘幼果实例分割训练。

输入图实例数量少、遮挡轻、背景简单的图片。请在类似的果园环境下，提高图像分割难度，但不要把它改成完全不同的场景。

柑橘必须是未成熟幼果。柑橘的纹理应该和参考图差不多，不要改变柑橘品种和纹理。

适当增加画面中的未成熟柑橘幼果数量，新增果实应自然分布在枝叶之间，以远处小目标、边缘小目标、半遮挡小目标为主。适当提高小目标占比，但不要生成无法辨认的噪点状果实。

适度增加真实叶片和细枝条遮挡，遮挡必须自然。

适当增加相邻果实靠近、局部重叠、边界接近或被叶片阴影干扰的情况，但不要把多个果实融合成畸形果实。

适当增加背景叶片密度、果叶颜色相近、局部阴影，使图像比参考图更难。

一定要真实一点，不要搞出很奇怪的光影
"""

NEGATIVE_PROMPT = (
    "mature orange fruit, ripe citrus, yellow orange fruit, apple, pear, tomato, non-citrus fruit, "
    "changed fruit texture, smooth plastic fruit skin, artificial fruit, deformed citrus, oversized fruit, "
    "fused fruits, unrealistic fruit cluster, cartoon, illustration, painting, CGI, 3D render, indoor scene, "
    "studio lighting, human hand, basket, text, watermark, logo, severe blur, unusable image, extreme occlusion, "
    "fully hidden fruit, unnatural branch, broken leaf artifacts, over-saturated color, unrealistic shadow"
)


def read_api_csv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                data[row[0].strip()] = row[1].strip()
    if not data.get("apiKey"):
        raise ValueError(f"apiKey not found in {path}")
    if not data.get("apiHost"):
        raise ValueError(f"apiHost not found in {path}")
    return data


def encode_image(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_image_urls(response_json: dict[str, Any]) -> list[str]:
    choices = response_json.get("output", {}).get("choices", [])
    urls: list[str] = []
    for choice in choices:
        content = choice.get("message", {}).get("content", [])
        for item in content:
            if item.get("image"):
                urls.append(item["image"])
    return urls


def request_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def download_image(url: str, output_path: Path) -> None:
    with request_session().get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input simple citrus image.")
    parser.add_argument("--api-csv", type=Path, default=DEFAULT_API_CSV, help="CSV containing apiKey and apiHost.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="qwen-image-2.0")
    parser.add_argument("--size", default="1152*2048", help="9:16 2K portrait size.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    creds = read_api_csv(args.api_csv)
    endpoint = f"https://{creds['apiHost'].rstrip('/')}/api/v1/services/aigc/multimodal-generation/generation"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": args.model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": encode_image(args.image)},
                        {"text": PROMPT},
                    ],
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": NEGATIVE_PROMPT,
            "prompt_extend": True,
            "size": args.size,
        },
    }
    if args.seed is not None:
        payload["parameters"]["seed"] = args.seed

    print(f"Input image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Output size: {args.size}")
    print("Submitting request...")

    response = request_session().post(
        endpoint,
        headers={
            "Authorization": f"Bearer {creds['apiKey']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=600,
    )

    try:
        response_json = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response: HTTP {response.status_code}, {response.text[:500]}") from exc

    if response.status_code != 200 or response_json.get("code"):
        message = response_json.get("message", response.text[:500])
        code = response_json.get("code", response.status_code)
        raise RuntimeError(f"Generation failed: {code}: {message}")

    urls = extract_image_urls(response_json)
    if not urls:
        raise RuntimeError(f"No image URL found in response: {json.dumps(response_json, ensure_ascii=False)[:1000]}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.image.stem}_qwen_hard_{stamp}"
    image_path = args.output_dir / f"{stem}.png"
    meta_path = args.output_dir / f"{stem}.json"

    print("Downloading generated image...")
    download_image(urls[0], image_path)

    metadata = {
        "source_image": str(args.image),
        "output_image": str(image_path),
        "model": args.model,
        "size": args.size,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "request_id": response_json.get("request_id"),
        "usage": response_json.get("usage"),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved image: {image_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
