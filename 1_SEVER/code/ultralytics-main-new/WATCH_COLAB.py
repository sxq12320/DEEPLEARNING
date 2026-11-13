"""Read live training progress and completed results from the Colab A100 session via HTTP REST."""

import base64
import subprocess
import sys
from pathlib import Path

UV_PYTHON = Path(r"C:\Users\33836\AppData\Roaming\uv\tools\google-colab-cli\Scripts\python.exe")

try:
    from colab_cli.common import state
    from colab_cli.contents import ContentsClient
except ImportError:
    if UV_PYTHON.is_file() and Path(sys.executable).resolve() != UV_PYTHON.resolve():
        raise SystemExit(subprocess.call([str(UV_PYTHON), __file__, *sys.argv[1:]]))
    raise


def read_remote_text(contents: ContentsClient, remote_path: str) -> str:
    data = contents._request("GET", remote_path, params={"content": "1"})
    raw = data.get("content", "")
    if data.get("format") == "base64":
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    return str(raw)


def main():
    sessions = state.store.list()
    sname = "citrus_l4" if "citrus_l4" in sessions else next(iter(sessions.keys()), "citrus_l4")
    s = state.get_session(sname)
    print(f"=== Connected to Colab Session: {sname} ({s.endpoint}, GPU={s.accelerator}) ===")
    contents = ContentsClient(s)
    proj = "content/results/PAPER_V12_VS78_SLICE_20261104"

    try:
        cur = read_remote_text(contents, f"{proj}/current_job.json")
        print("=== CURRENT JOB ===")
        print(cur.strip())
    except Exception:
        pass

    try:
        items = contents.list_dir(proj).get("content", [])
        for item in sorted(items, key=lambda x: x.get("name", "")):
            name = item.get("name", "")
            if item.get("type") == "directory" and name.startswith("P"):
                try:
                    csv_text = read_remote_text(contents, f"{proj}/{name}/results.csv")
                    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
                    if len(lines) > 1:
                        data_rows = lines[1:]
                        print(f"\n=== {name} ({len(data_rows)}/300 epochs completed) ===")
                        headers = [h.strip() for h in lines[0].split(",")]
                        for row in data_rows[-5:]:
                            vals = [v.strip() for v in row.split(",")]
                            kv = dict(zip(headers, vals))
                            print(
                                f"  Epoch {kv.get('epoch')}: "
                                f"Mask mAP50={kv.get('metrics/mAP50(M)')}, "
                                f"Mask mAP50-95={kv.get('metrics/mAP50-95(M)')}, "
                                f"Box mAP50={kv.get('metrics/mAP50(B)')}, "
                                f"Recall(M)={kv.get('metrics/recall(M)')}, "
                                f"time={kv.get('time')}s"
                            )
                except Exception:
                    pass
    except Exception:
        pass

    for summary_file in ("paper_results.csv", "P00_P01_COMPARISON.csv", "P00_SLICE_COMPARISON.csv"):
        try:
            txt = read_remote_text(contents, f"{proj}/{summary_file}")
            print(f"\n=== {summary_file} ===")
            print(txt.strip())
        except Exception:
            pass

    try:
        log_text = read_remote_text(contents, "content/train_ablation.log")
        lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
        print("\n=== LATEST 15 LOG LINES ===")
        for ln in lines[-15:]:
            print(ln.encode("ascii", errors="ignore").decode("ascii"))
    except Exception as e:
        print("Log read error:", e)


if __name__ == "__main__":
    main()
