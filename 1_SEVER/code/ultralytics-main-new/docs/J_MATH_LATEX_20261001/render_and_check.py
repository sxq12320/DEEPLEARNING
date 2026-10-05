"""Render every final LaTeX page and record geometric and textual QA.

This does not replace inspection of every rendered page by the author.
"""

import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PDF = HERE / "J11_math_derivation.pdf"
PREVIEWS = HERE / "qa" / "pages"


def main():
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    document = fitz.open(PDF)
    checks, thumbnails = [], []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
    all_text = []
    for index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.55, 1.55), alpha=False)
        target = PREVIEWS / f"page_{index + 1:02d}.png"
        pixmap.save(target)
        text = page.get_text()
        all_text.append(text)
        invalid = []
        smallest = float("inf")
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    smallest = min(smallest, span["size"])
                    rect = fitz.Rect(span["bbox"])
                    if (
                        rect.x0 < -0.5
                        or rect.y0 < -0.5
                        or rect.x1 > page.rect.width + 0.5
                        or rect.y1 > page.rect.height + 0.5
                    ):
                        invalid.append(span["text"])
        checks.append(
            {
                "pdf_page": index + 1,
                "text_characters": len(text),
                "smallest_span_pt": round(smallest, 3),
                "outside_page": invalid,
                "replacement_character": "\ufffd" in text,
                "preview": str(target),
            }
        )
        image = Image.open(target).convert("RGB")
        image.thumbnail((440, 625))
        tile = Image.new("RGB", (460, 660), "#e8edf2")
        tile.paste(image, ((460 - image.width) // 2, 28))
        ImageDraw.Draw(tile).text((12, 2), f"PDF page {index + 1}", font=font, fill="#20374b")
        thumbnails.append(tile)
    for start in range(0, len(thumbnails), 6):
        sheet = Image.new("RGB", (460 * 3, 660 * 2), "#d7e0e7")
        for offset, tile in enumerate(thumbnails[start : start + 6]):
            sheet.paste(tile, ((offset % 3) * 460, (offset // 3) * 660))
        sheet.save(HERE / "qa" / f"contact_{start // 6 + 1:02d}.png")
    # Two-page sheets preserve the original ~920 px page width for visual QA.
    for start in range(0, len(checks), 2):
        files = [Path(item["preview"]) for item in checks[start : start + 2]]
        images = [Image.open(item).convert("RGB") for item in files]
        sheet = Image.new("RGB", (sum(item.width for item in images), max(item.height for item in images)), "white")
        cursor = 0
        for image in images:
            sheet.paste(image, (cursor, 0))
            cursor += image.width
        sheet.save(HERE / "qa" / f"page_pair_{start // 2 + 1:02d}.png")
    assert all(not row["outside_page"] for row in checks)
    assert all(not row["replacement_character"] for row in checks)
    text = "\n".join(all_text)
    assert "J11" in text and "差分" in text and "证明" in text
    record = {
        "pages": len(document),
        "render_scale": 1.55,
        "source": str(PDF),
        "page_checks": checks,
        "visual_review": "Requires manual review of every final page; contact sheets are navigation aids.",
    }
    (HERE / "qa" / "pdf_check.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "qa" / "extracted_text.txt").write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "pages": len(document),
                "out_of_page_spans": 0,
                "replacement_characters": 0,
                "min_span_pt": min(row["smallest_span_pt"] for row in checks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
