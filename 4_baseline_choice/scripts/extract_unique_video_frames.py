"""Extract visually distinct frames from one or more videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


ANALYSIS_FIELDS = [
    "source_video",
    "frame_index",
    "timestamp_seconds",
    "retained",
    "decision",
    "output_image",
    "nearest_kept_image",
    "nearest_phash_distance",
    "max_ssim_to_kept",
    "ssim_to_previous_frame",
    "mean_absdiff_to_previous",
    "changed_pixel_ratio_previous",
    "sharpness",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_frequency = dct[:8, :8]
    median = np.median(low_frequency.flatten()[1:])
    return low_frequency > median


def hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def analysis_gray(frame: np.ndarray, size: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute grayscale SSIM without requiring an additional package."""
    left_float = left.astype(np.float64)
    right_float = right.astype(np.float64)
    constant_1 = (0.01 * 255) ** 2
    constant_2 = (0.03 * 255) ** 2

    mean_left = cv2.GaussianBlur(left_float, (11, 11), 1.5)
    mean_right = cv2.GaussianBlur(right_float, (11, 11), 1.5)
    variance_left = cv2.GaussianBlur(left_float * left_float, (11, 11), 1.5) - mean_left**2
    variance_right = cv2.GaussianBlur(right_float * right_float, (11, 11), 1.5) - mean_right**2
    covariance = cv2.GaussianBlur(left_float * right_float, (11, 11), 1.5) - mean_left * mean_right

    numerator = (2 * mean_left * mean_right + constant_1) * (2 * covariance + constant_2)
    denominator = (mean_left**2 + mean_right**2 + constant_1) * (
        variance_left + variance_right + constant_2
    )
    score = float(np.mean(numerator / np.maximum(denominator, np.finfo(np.float64).eps)))
    return float(np.clip(score, -1.0, 1.0))


def frame_difference(current: np.ndarray, previous: np.ndarray) -> tuple[float, float]:
    difference = cv2.absdiff(current, previous)
    mean_difference = float(np.mean(difference))
    changed_ratio = float(np.count_nonzero(difference >= 20) / difference.size)
    return mean_difference, changed_ratio


def candidate_frames(
    capture: cv2.VideoCapture,
    fps: float,
    frame_count: int,
    interval: float,
    every_frame: bool,
):
    if every_frame:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps if fps > 0 else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            yield frame_index, timestamp, frame
            frame_index += 1
        return

    duration = frame_count / fps if fps > 0 else 0.0
    timestamp = interval / 2
    while timestamp < duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if ok:
            frame_index = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            yield frame_index, timestamp, frame
        timestamp += interval


def print_progress(current: int, total: int, kept: int) -> None:
    if total <= 0:
        print(f"\r  scanned {current} frames | kept {kept}", end="", flush=True)
        return
    width = 30
    ratio = min(1.0, current / total)
    completed = int(width * ratio)
    bar = "#" * completed + "-" * (width - completed)
    print(f"\r  [{bar}] {ratio:6.1%} | {current}/{total} | kept {kept}", end="", flush=True)


def unique_videos(paths: list[Path]) -> tuple[list[Path], list[tuple[Path, Path]]]:
    unique: list[Path] = []
    duplicate_pairs: list[tuple[Path, Path]] = []
    hashes: dict[str, Path] = {}
    for path in paths:
        digest = file_sha256(path)
        if digest in hashes:
            duplicate_pairs.append((path, hashes[digest]))
        else:
            hashes[digest] = path
            unique.append(path)
    return unique, duplicate_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5, help="Candidate-frame interval in seconds.")
    parser.add_argument(
        "--every-frame",
        action="store_true",
        help="Decode and analyze every original video frame instead of sampling by time.",
    )
    parser.add_argument(
        "--min-phash-distance",
        type=int,
        default=10,
        help="Reject a frame when its nearest retained-frame pHash distance is below this value.",
    )
    parser.add_argument(
        "--max-ssim",
        type=float,
        default=0.92,
        help="Reject a frame when SSIM to a retained frame reaches this value.",
    )
    parser.add_argument(
        "--ssim-candidates",
        type=int,
        default=3,
        help="Number of pHash-nearest retained frames checked with SSIM.",
    )
    parser.add_argument("--comparison-size", type=int, default=320)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    missing = [path for path in args.videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Video not found: {missing[0]}")
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")
    if not 0 < args.max_ssim <= 1:
        raise ValueError("--max-ssim must be in the range (0, 1].")
    if args.ssim_candidates <= 0:
        raise ValueError("--ssim-candidates must be positive.")
    if args.comparison_size < 32:
        raise ValueError("--comparison-size must be at least 32.")

    videos, duplicates = unique_videos(args.videos)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "frames_manifest.csv"
    analysis_path = args.output_dir / "all_frames_analysis.csv"
    summary_path = args.output_dir / "video_summary.csv"

    kept_frames: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    kept_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    candidate_count = 0

    for duplicate, original in duplicates:
        print(f"Skip exact duplicate: {duplicate.name} == {original.name}")
        summary_records.append(
            {
                "source_video": str(duplicate),
                "status": "exact_file_duplicate",
                "duplicate_of": str(original),
                "decoded_frames": 0,
                "retained_frames": 0,
                "rejected_frames": 0,
            }
        )

    for video_index, video_path in enumerate(videos, start=1):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        expected_candidates = frame_count if args.every_frame else max(1, int(np.ceil(duration / args.interval)))
        mode = "every frame" if args.every_frame else f"every {args.interval:g}s"
        print(f"[{video_index}/{len(videos)}] {video_path.name}: {duration:.2f}s, {fps:.2f} FPS, {mode}")

        decoded_in_video = 0
        kept_in_video = 0
        previous_gray: np.ndarray | None = None
        for frame_index, timestamp, frame in candidate_frames(
            capture,
            fps,
            frame_count,
            args.interval,
            args.every_frame,
        ):
            candidate_count += 1
            decoded_in_video += 1
            gray = analysis_gray(frame, args.comparison_size)
            frame_hash = perceptual_hash(gray)
            distances = [
                hamming_distance(frame_hash, saved["phash"])
                for saved in kept_frames
            ]
            nearest_distance = min(distances) if distances else 64
            nearest_image = ""
            max_ssim: float | None = None

            if kept_frames:
                ranked_indices = sorted(range(len(distances)), key=distances.__getitem__)
                nearest_image = str(kept_frames[ranked_indices[0]]["image"])
                nearest_indices = ranked_indices[: args.ssim_candidates]
                for kept_index in nearest_indices:
                    score = structural_similarity(gray, kept_frames[kept_index]["gray"])
                    if max_ssim is None or score > max_ssim:
                        max_ssim = score

            if previous_gray is None:
                previous_ssim = None
                mean_difference = None
                changed_ratio = None
            else:
                previous_ssim = structural_similarity(gray, previous_gray)
                mean_difference, changed_ratio = frame_difference(gray, previous_gray)

            if not kept_frames:
                retained = True
                decision = "first_frame"
            elif nearest_distance < args.min_phash_distance:
                retained = False
                decision = "near_duplicate_phash"
            elif max_ssim is not None and max_ssim >= args.max_ssim:
                retained = False
                decision = "near_duplicate_ssim"
            else:
                retained = True
                decision = "different"

            output_name = ""
            if retained:
                output_name = (
                    f"video{video_index:02d}_f{frame_index + 1:06d}"
                    f"_t{round(timestamp * 1000):08d}ms.jpg"
                )
                output_path = args.output_dir / output_name
                if not cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                ):
                    raise RuntimeError(f"Failed to write frame: {output_path}")
                kept_frames.append({"phash": frame_hash, "gray": gray.copy(), "image": output_name})
                kept_in_video += 1

            full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(full_gray, cv2.CV_64F).var())
            record = {
                "source_video": str(video_path),
                "frame_index": frame_index + 1,
                "timestamp_seconds": f"{timestamp:.3f}",
                "retained": int(retained),
                "decision": decision,
                "output_image": output_name,
                "nearest_kept_image": nearest_image,
                "nearest_phash_distance": nearest_distance,
                "max_ssim_to_kept": "" if max_ssim is None else f"{max_ssim:.6f}",
                "ssim_to_previous_frame": "" if previous_ssim is None else f"{previous_ssim:.6f}",
                "mean_absdiff_to_previous": "" if mean_difference is None else f"{mean_difference:.4f}",
                "changed_pixel_ratio_previous": "" if changed_ratio is None else f"{changed_ratio:.6f}",
                "sharpness": f"{sharpness:.2f}",
            }
            records.append(record)
            if retained:
                kept_records.append(record)

            previous_gray = gray
            if decoded_in_video == expected_candidates or decoded_in_video % max(1, int(fps)) == 0:
                print_progress(decoded_in_video, expected_candidates, kept_in_video)

        capture.release()
        print_progress(decoded_in_video, expected_candidates, kept_in_video)
        print()
        summary_records.append(
            {
                "source_video": str(video_path),
                "status": "processed",
                "duplicate_of": "",
                "decoded_frames": decoded_in_video,
                "retained_frames": kept_in_video,
                "rejected_frames": decoded_in_video - kept_in_video,
            }
        )

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ANALYSIS_FIELDS)
        writer.writeheader()
        writer.writerows(kept_records)

    with analysis_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ANALYSIS_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_video",
                "status",
                "duplicate_of",
                "decoded_frames",
                "retained_frames",
                "rejected_frames",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_records)

    print(f"Candidate frames: {candidate_count}")
    print(f"Unique frames: {len(kept_records)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Every-frame analysis: {analysis_path}")
    print(f"Video summary: {summary_path}")


if __name__ == "__main__":
    main()
