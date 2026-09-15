"""Read all historical results without overwriting earlier evidence reports."""

import json

import review_citrus_ev7_for_v8 as review


def main():
    review.OUT = review.ROOT / "docs/E_V9_REVIEW_20260915"
    review.main()
    path = review.OUT / "audit.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["v8"] = [r for r in data["history"] if "/E/E_V8/" in r["path"].replace("\\", "/")]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in data["v8"]:
        print(
            r["name"],
            r["epochs"],
            "AP50/AP/last20",
            *[round(v * 100, 3) for v in (r["best"]["metrics/mAP50(M)"], r["best"][review.AP], r["last20"])],
            "sec",
            r["median_epoch_s"],
        )
        for budget, d in r["paired"].items():
            for mode in ("global", "trustedmask"):
                s = d["summary"][mode]
                print(
                    budget,
                    mode,
                    "AP50/AP",
                    round(s["metrics/mAP50(M)"] * 100, 3),
                    round(s[review.AP] * 100, 3),
                    "tiny",
                    s["tiny_matched"],
                    "bg",
                    s["errors25"]["background"],
                    "P90R",
                    round(s["operating_p90"]["recall"] * 100, 3),
                    "ms",
                    round(s["median_ms"], 2),
                )


if __name__ == "__main__":
    main()
