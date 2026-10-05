# -*- coding: utf-8 -*-
"""Synthesize every narration via edge-tts -> _build/audio/<L>/<scene>.mp3."""
import asyncio
import json
import os

import edge_tts

ROOT = os.path.dirname(os.path.abspath(__file__))
VOICE = "zh-CN-YunjianNeural"
OUT = os.path.join(ROOT, "_build", "audio")

narr = json.load(open(os.path.join(ROOT, "narrations.json"), encoding="utf-8"))


async def one(sem, lesson, scene, text):
    path = os.path.join(OUT, lesson, scene + ".mp3")
    if os.path.exists(path) and os.path.getsize(path) > 2000:
        return
    async with sem:
        for attempt in range(4):
            try:
                comm = edge_tts.Communicate(text, VOICE)
                await comm.save(path)
                print(f"{lesson}/{scene} {os.path.getsize(path)//1024}KB",
                      flush=True)
                return
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"retry {lesson}/{scene}: {e}", flush=True)
                await asyncio.sleep(2 + attempt * 3)


async def main():
    sem = asyncio.Semaphore(3)
    jobs = [
        one(sem, lesson, scene, text)
        for lesson, scenes in narr.items()
        for scene, text in scenes.items()
    ]
    for lesson in narr:
        os.makedirs(os.path.join(OUT, lesson), exist_ok=True)
    await asyncio.gather(*jobs)


asyncio.run(main())
print("TTS ALL DONE")
