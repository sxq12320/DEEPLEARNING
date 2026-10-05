"""Build episode assets: per-section TTS mp3 + sentence-level timing -> timeline.json.

Usage:
    python tools/make_episode.py 024

Reads  episodes/<id>/script.md
Writes public/audio/<id>/s{i}.mp3 and src/timeline_<id>.json
"""
import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
VOICE = "zh-CN-YunjianNeural"
TITLE_CARD_SEC = 3.0
END_CARD_SEC = 2.5
SECTION_PAD_SEC = 0.6  # pause appended after each section's audio

SENT_END = "。！？!?"


def probe_duration(mp3: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


async def synth(text: str, mp3_path: Path):
    comm = edge_tts.Communicate(text, VOICE)
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                words.append(
                    (chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"])
                )
    return words


def words_to_sentences(words):
    """Group word-boundary events into sentence cues split on end punctuation."""
    cues = []
    buf, t0 = [], None
    for start, dur, w in words:
        if t0 is None:
            t0 = start
        buf.append(w)
        if any(w.rstrip().endswith(p) for p in SENT_END):
            cues.append({"t0": round(t0, 3), "t1": round(start + dur, 3), "text": "".join(buf)})
            buf, t0 = [], None
    if buf:
        cues.append({"t0": round(t0, 3), "t1": round(words[-1][0] + words[-1][1], 3), "text": "".join(buf)})
    return cues


def parse_script(md: str):
    title = re.search(r"^#\s+(.+)$", md, flags=re.M).group(1).strip()
    sections = []
    for m in re.finditer(r"^##\s+([^|\n]+?)(?:\|\s*([^\n]+))?\n+([^#]+?)(?=^##|\Z)", md, flags=re.M):
        name = m.group(1).strip()
        keywords = [k.strip() for k in (m.group(2) or "").split(",") if k.strip()]
        text = re.sub(r"\s+", "", m.group(3))
        sections.append({"name": name, "keywords": keywords, "text": text})
    return title, sections


async def main(ep_id: str):
    md = (ROOT / "episodes" / ep_id / "script.md").read_text(encoding="utf-8")
    title, sections = parse_script(md)
    audio_dir = ROOT / "public" / "audio" / ep_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    cursor = TITLE_CARD_SEC
    out_sections = []
    for i, sec in enumerate(sections):
        mp3 = audio_dir / f"s{i}.mp3"
        words = await synth(sec["text"], mp3)
        dur = probe_duration(mp3)
        cues = words_to_sentences(words)
        for c in cues:  # make cue times absolute
            c["t0"] += cursor
            c["t1"] += cursor
        out_sections.append({
            **sec,
            "audio": f"audio/{ep_id}/s{i}.mp3",
            "start": round(cursor, 3),
            "dur": round(dur + SECTION_PAD_SEC, 3),
            "cues": cues,
        })
        cursor += dur + SECTION_PAD_SEC

    timeline = {
        "title": title,
        "episode": ep_id,
        "titleCardSec": TITLE_CARD_SEC,
        "endCardSec": END_CARD_SEC,
        "totalSec": round(cursor + END_CARD_SEC, 3),
        "sections": out_sections,
    }
    out = ROOT / "src" / f"timeline_{ep_id}.json"
    out.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK {out} total={timeline['totalSec']}s sections={len(out_sections)}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
