# -*- coding: utf-8 -*-
"""Paced Semantic Scholar search for citrus-seg improvement lit review.
Single process, internal rate-limiting + 429 backoff. Prints compact records.
"""
import json
import sys
import time
import urllib.parse
import urllib.request

API_KEY = "s2k-mMUmGmVGTGFsILLZitJOlX57ybOT3Yutbmb9R36O"
BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,venue,externalIds,abstract,citationCount,url"

# (dimension, query, limit)
QUERIES = [
    # A. instance segmentation base methods
    ("BASE", "YOLACT real-time instance segmentation", 8),
    ("BASE", "SOLOv2 dynamic instance segmentation", 6),
    ("BASE", "YOLO instance segmentation real time", 8),
    # B. lightweight backbone
    ("BACKBONE", "lightweight backbone efficient object detection", 10),
    ("BACKBONE", "efficient CNN architecture mobile detection", 8),
    ("BACKBONE", "StarNet star operation network", 6),
    # C. attention
    ("ATTN", "coordinate attention efficient mobile network", 10),
    ("ATTN", "efficient channel attention ECA", 8),
    ("ATTN", "CBAM convolutional block attention module", 8),
    ("ATTN", "EMA efficient multiscale attention cross spatial", 6),
    # D. feature fusion / neck
    ("NECK", "asymptotic feature pyramid network AFPN detection", 6),
    ("NECK", "attentional scale sequence fusion instance segmentation", 6),
    ("NECK", "BiFPN weighted bidirectional feature pyramid", 6),
    ("NECK", "gather distribute mechanism Gold-YOLO", 6),
    # E. frequency / texture / camouflage (NOVELTY: green fruit in green leaves)
    ("FREQ", "frequency domain learning convolutional neural network", 10),
    ("FREQ", "wavelet transform CNN feature extraction", 8),
    ("FREQ", "camouflaged object detection frequency", 10),
    ("FREQ", "texture feature deep learning fine-grained", 8),
    ("FREQ", "high frequency detail segmentation boundary", 6),
    # F. small object
    ("SMALL", "small object detection feature pyramid survey", 8),
    ("SMALL", "tiny object detection high resolution feature", 8),
    # G. loss / augmentation / assignment
    ("LOSS", "copy paste augmentation instance segmentation", 8),
    ("LOSS", "boundary aware loss segmentation", 8),
    ("LOSS", "focal loss dense object detection", 6),
    ("LOSS", "task aligned assigner one stage detection", 6),
    ("LOSS", "Wise-IoU bounding box regression loss", 6),
    # H. agricultural domain
    ("DOMAIN", "green immature fruit detection deep learning orchard", 10),
    ("DOMAIN", "citrus fruit detection segmentation convolutional", 10),
    ("DOMAIN", "occluded fruit instance segmentation harvesting", 8),
    ("DOMAIN", "green citrus detection yield estimation", 8),
]


def fetch(query, limit, retries=5):
    params = urllib.parse.urlencode({"query": query, "limit": limit, "fields": FIELDS})
    url = f"{BASE}?{params}"
    req = urllib.request.Request(url, headers={"x-api-key": API_KEY})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [429] backoff {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  [HTTP {e.code}] {query}", file=sys.stderr)
            return None
        except Exception as e:  # noqa
            print(f"  [ERR] {query}: {e}", file=sys.stderr)
            time.sleep(4)
    return None


def ext_id(ext):
    if not ext:
        return "-"
    if ext.get("DOI"):
        return "DOI:" + ext["DOI"]
    if ext.get("ArXiv"):
        return "arXiv:" + ext["ArXiv"]
    for k in ("PubMed", "MAG", "CorpusId"):
        if ext.get(k):
            return f"{k}:{ext[k]}"
    return "-"


def main():
    seen = set()
    for dim, query, limit in QUERIES:
        print(f"\n===== [{dim}] {query} =====")
        data = fetch(query, limit)
        if not data or "data" not in data:
            print("  (no data)")
            time.sleep(3.5)
            continue
        for p in data["data"]:
            pid = p.get("paperId")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            authors = p.get("authors") or []
            first = authors[0]["name"] if authors else "?"
            etal = " et al." if len(authors) > 1 else ""
            venue = p.get("venue") or "preprint"
            year = p.get("year") or "?"
            cites = p.get("citationCount", 0)
            eid = ext_id(p.get("externalIds"))
            abs = (p.get("abstract") or "").replace("\n", " ")
            if len(abs) > 300:
                abs = abs[:300] + "..."
            print(f"- TITLE: {p.get('title')}")
            print(f"  AUTH: {first}{etal} | {year} | {venue} | cites={cites}")
            print(f"  ID: {eid} | pid={pid}")
            print(f"  URL: {p.get('url')}")
            print(f"  ABS: {abs}")
        time.sleep(3.5)
    print(f"\n===== DONE. unique papers: {len(seen)} =====")


if __name__ == "__main__":
    main()
