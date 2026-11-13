"""Create the paired min8 dataset from the actual server source, then start the E12 queue."""
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import RUN_E12_ALL_LABEL8_COMPARE as suite
from RUN_ES03_LABEL8_COMPARE import audit_labels
from RUN_LABEL8_COMPARE import dataset_configs

# Source used in the pasted server log. Keep the existing suite's model/GPU/epoch settings.
OLD_DATA = "/data/sxq/datasets/orange_yolo/data.yaml"


def prepare(old_data):
    source_yaml = Path(old_data).expanduser().resolve()
    if not source_yaml.is_file():
        raise FileNotFoundError(source_yaml)
    source = source_yaml.parent
    target = source.with_name(source.name + "_min8px640")
    if target.exists():
        report = target / "cleaning_report.json"
        if not report.is_file() or Path(json.loads(report.read_text(encoding="utf-8"))["source"]).resolve() != source:
            raise FileExistsError(f"Existing directory has no matching source record: {target}")
        print(f"Reusing {target}; checking all three splits", flush=True)
        audit_labels(dataset_configs(source_yaml, target / 'data.yaml', True))
    else:
        # An interrupted copy cannot become a published dataset or block the next invocation.
        with TemporaryDirectory(prefix=target.name + "_staging_", dir=str(source.parent)) as tmp:
            candidate = Path(tmp) / "dataset"
            subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("prepare_orange_grouped_clean_minpx.py")),
                            "--source", str(source), "--out", str(candidate), "--imgsz", "640", "--min-side", "8",
                            "--clean-splits", "train,val,test"], check=True)
            audit_labels(dataset_configs(source_yaml, candidate / "data.yaml", True))
            candidate.rename(target)
    print(f"Verified filtered YAML: {target / 'data.yaml'}", flush=True)
    return target / "data.yaml"


def main():
    suite.OLD_DATA = OLD_DATA
    suite.NEW_DATA = str(prepare(OLD_DATA))
    suite.main()


if __name__ == "__main__":
    main()
