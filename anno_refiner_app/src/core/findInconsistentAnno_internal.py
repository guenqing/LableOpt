#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class YoloBox:
    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        x1 = self.xc - self.w / 2.0
        y1 = self.yc - self.h / 2.0
        x2 = self.xc + self.w / 2.0
        y2 = self.yc + self.h / 2.0
        return x1, y1, x2, y2

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class FrameAnn:
    video_rel: Path
    frame_num: int
    txt_path: Path
    jpg_path: Path
    boxes: list[YoloBox]
    focus_boxes: list[YoloBox]
    dhash64: Optional[int]
    img_missing: bool
    ann_parse_error: Optional[str]
    img_error: Optional[str]

    @property
    def has_focus(self) -> bool:
        return bool(self.focus_boxes)


def parse_frame_number(filename: str) -> int:
    stem = Path(filename).stem
    if not stem.startswith("frame_"):
        raise ValueError(f"Unexpected frame filename: {filename}")
    num_s = stem[len("frame_") :]
    if not num_s.isdigit():
        raise ValueError(f"Unexpected frame filename: {filename}")
    return int(num_s)


def parse_yolo_lines(text: str) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for line_idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Bad YOLO line (line {line_idx}): {raw!r}")
        cls_s, xc_s, yc_s, w_s, h_s = parts
        try:
            cls = int(cls_s)
            xc = float(xc_s)
            yc = float(yc_s)
            w = float(w_s)
            h = float(h_s)
        except ValueError as e:
            raise ValueError(f"Bad YOLO line (line {line_idx}): {raw!r}") from e
        boxes.append(YoloBox(cls=cls, xc=xc, yc=yc, w=w, h=h))
    return boxes


def load_yolo_txt(path: Path) -> list[YoloBox]:
    return parse_yolo_lines(path.read_text(encoding="utf-8", errors="replace"))


def box_iou(a: YoloBox, b: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = a.to_xyxy()
    bx1, by1, bx2, by2 = b.to_xyxy()

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def max_iou(boxes_a: Iterable[YoloBox], boxes_b: Iterable[YoloBox]) -> float:
    best = 0.0
    boxes_b_list = list(boxes_b)
    for a in boxes_a:
        for b in boxes_b_list:
            best = max(best, box_iou(a, b))
    return best


def dhash_64_from_image_path(image_path: Path) -> int:
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PIL (Pillow) is required for dhash. Please install pillow.") from e

    with Image.open(image_path) as img:
        img = img.convert("L").resize((9, 8))
        pixels = list(img.getdata())

    bits = 0
    for row in range(8):
        off = row * 9
        for col in range(8):
            left = pixels[off + col]
            right = pixels[off + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming_distance_64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _compute_dhash_worker(jpg_path_str: str) -> tuple[str, Optional[int], Optional[str]]:
    """Worker function for multiprocessing dHash computation.
    
    Returns: (jpg_path_str, dhash64 or None, error_msg or None)
    """
    try:
        dhash64 = dhash_64_from_image_path(Path(jpg_path_str))
        return (jpg_path_str, dhash64, None)
    except Exception as e:
        return (jpg_path_str, None, repr(e))


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad(values: list[float], med: float) -> Optional[float]:
    if not values:
        return None
    dev = [abs(v - med) for v in values]
    return _median(dev)


def load_dhash_cache(cache_path: Path) -> dict[str, int]:
    if not cache_path.exists():
        return {}
    cache: dict[str, int] = {}
    with cache_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            rel, hex_s = row[0], row[1].strip()
            if not rel or not hex_s:
                continue
            try:
                cache[rel] = int(hex_s, 16)
            except ValueError:
                continue
    logger.info("Loaded dhash cache: %s (%d entries)", cache_path, len(cache))
    return cache


def save_dhash_cache(cache: dict[str, int], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_jpg_path", "dhash64_hex"])
        for rel in sorted(cache.keys()):
            writer.writerow([rel, f"{cache[rel]:016x}"])
    logger.info("Saved dhash cache: %s (%d entries)", cache_path, len(cache))


def scan_annotations(
    *,
    frames_root: Path,
    ann_root: Path,
    focus_cls: int,
    compute_hash: bool,
    dhash_cache: Optional[dict[str, int]],
    max_txt: int,
    log_every: int,
    num_workers: int,
) -> tuple[list[FrameAnn], dict[str, Any]]:
    txt_paths = sorted(ann_root.rglob("frame_*.txt"))
    if max_txt > 0:
        txt_paths = txt_paths[:max_txt]

    meta: dict[str, Any] = {
        "ann_root": str(ann_root),
        "frames_root": str(frames_root),
        "txt_count": len(txt_paths),
    }

    # Phase 1: Parse annotations and collect paths needing hash computation
    records: list[FrameAnn] = []
    missing_images = 0
    parse_errors = 0
    cached_hash = 0
    
    # Track which records need hash computation
    needs_hash: list[tuple[int, str, Path]] = []  # (record_idx, rel_jpg, jpg_path)

    for idx, txt_path in enumerate(txt_paths, start=1):
        if log_every > 0 and idx % log_every == 0:
            logger.info("Scanning annotations (phase 1): %d/%d", idx, len(txt_paths))

        rel = txt_path.relative_to(ann_root)
        jpg_path = (frames_root / rel).with_suffix(".jpg")
        video_rel = rel.parent
        frame_num = parse_frame_number(txt_path.name)

        boxes: list[YoloBox] = []
        ann_parse_error: Optional[str] = None
        try:
            boxes = load_yolo_txt(txt_path)
        except Exception as e:
            ann_parse_error = repr(e)
            parse_errors += 1

        focus_boxes = [b for b in boxes if b.cls == focus_cls]

        img_missing = not jpg_path.exists()
        if img_missing:
            missing_images += 1

        dhash64: Optional[int] = None
        if compute_hash and not img_missing:
            rel_jpg = str(jpg_path.relative_to(frames_root))
            if dhash_cache is not None and rel_jpg in dhash_cache:
                dhash64 = dhash_cache[rel_jpg]
                cached_hash += 1
            else:
                # Mark for parallel computation
                needs_hash.append((len(records), rel_jpg, jpg_path))

        records.append(
            FrameAnn(
                video_rel=video_rel,
                frame_num=frame_num,
                txt_path=txt_path,
                jpg_path=jpg_path,
                boxes=boxes,
                focus_boxes=focus_boxes,
                dhash64=dhash64,
                img_missing=img_missing,
                ann_parse_error=ann_parse_error,
                img_error=None,
            )
        )

    # Phase 2: Parallel dHash computation for uncached images
    computed_hash = 0
    hash_errors = 0
    
    if needs_hash:
        logger.info(
            "Computing dHash for %d images using %d workers...",
            len(needs_hash), num_workers
        )
        
        # Prepare paths for parallel processing
        jpg_paths_to_compute = [str(item[2]) for item in needs_hash]
        
        if num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_compute_dhash_worker, jpg_paths_to_compute)
        else:
            # Single-threaded fallback
            results = [_compute_dhash_worker(p) for p in jpg_paths_to_compute]
        
        # Update records with computed hashes
        for (record_idx, rel_jpg, _), (_, dhash64, error) in zip(needs_hash, results):
            if error is not None:
                records[record_idx].img_error = error
                hash_errors += 1
            else:
                records[record_idx].dhash64 = dhash64
                computed_hash += 1
                if dhash_cache is not None:
                    dhash_cache[rel_jpg] = dhash64
        
        logger.info("dHash computation done: computed=%d, errors=%d", computed_hash, hash_errors)

    meta.update(
        {
            "missing_images": missing_images,
            "ann_parse_errors": parse_errors,
            "hash_errors": hash_errors,
            "computed_hash": computed_hash,
            "cached_hash": cached_hash,
        }
    )
    return records, meta


def analyze_inconsistent_annotations(
    *,
    frames_root: Path,
    ann_root: Path,
    focus_cls: int,
    compute_hash: bool,
    max_pair_frame_gap: int,
    max_hash_dist: int,
    neighbor_min_iou: float,
    triad_require_iou: float,
    isolated_window: int,
    outlier_mad_k: float,
    min_score: int,
    max_txt: int,
    dhash_cache: Optional[dict[str, int]],
    num_workers: int,
) -> dict[str, Any]:
    records, scan_meta = scan_annotations(
        frames_root=frames_root,
        ann_root=ann_root,
        focus_cls=focus_cls,
        compute_hash=compute_hash,
        dhash_cache=dhash_cache,
        max_txt=max_txt,
        log_every=1000,
        num_workers=num_workers,
    )

    by_video: dict[Path, list[FrameAnn]] = {}
    for r in records:
        by_video.setdefault(r.video_rel, []).append(r)
    for v in by_video:
        by_video[v].sort(key=lambda x: x.frame_num)

    rec_map: dict[tuple[Path, int], FrameAnn] = {(r.video_rel, r.frame_num): r for r in records}

    class_line_counts: dict[int, int] = {}
    frames_with_focus = 0
    for r in records:
        if r.has_focus:
            frames_with_focus += 1
        for b in r.boxes:
            class_line_counts[b.cls] = class_line_counts.get(b.cls, 0) + 1

    suspicious: dict[tuple[str, int], dict[str, Any]] = {}

    def add_susp(video_rel: Path, frame_num: int, *, reason: str, score: int) -> None:
        key = (str(video_rel), frame_num)
        item = suspicious.setdefault(
            key,
            {
                "video_rel": str(video_rel),
                "frame_num": frame_num,
                "score": 0,
                "reasons": set(),
            },
        )
        item["score"] += score
        item["reasons"].add(reason)

    # Per video heuristics
    for video_rel, frames in by_video.items():
        # Basic file issues
        for r in frames:
            if r.img_missing:
                add_susp(video_rel, r.frame_num, reason="missing_corresponding_jpg", score=2)
            if r.ann_parse_error is not None:
                add_susp(video_rel, r.frame_num, reason="annotation_parse_error", score=3)
            if r.img_error is not None:
                add_susp(video_rel, r.frame_num, reason="image_hash_error", score=2)

        if compute_hash:
            # Neighbor pair rules
            for prev, curr in zip(frames, frames[1:]):
                gap = curr.frame_num - prev.frame_num
                if gap <= 0 or gap > max_pair_frame_gap:
                    continue
                if prev.dhash64 is None or curr.dhash64 is None:
                    continue
                dist = hamming_distance_64(prev.dhash64, curr.dhash64)
                if dist > max_hash_dist:
                    continue

                if prev.has_focus != curr.has_focus:
                    add_susp(
                        video_rel,
                        prev.frame_num,
                        reason=f"neighbor_similar_but_focus_diff(dist={dist},gap={gap})",
                        score=3,
                    )
                    add_susp(
                        video_rel,
                        curr.frame_num,
                        reason=f"neighbor_similar_but_focus_diff(dist={dist},gap={gap})",
                        score=3,
                    )
                elif prev.has_focus and curr.has_focus:
                    iou = max_iou(prev.focus_boxes, curr.focus_boxes)
                    if iou < neighbor_min_iou:
                        add_susp(
                            video_rel,
                            prev.frame_num,
                            reason=(
                                f"neighbor_similar_but_box_mismatch(iou={iou:.3f},dist={dist},gap={gap})"
                            ),
                            score=2,
                        )
                        add_susp(
                            video_rel,
                            curr.frame_num,
                            reason=(
                                f"neighbor_similar_but_box_mismatch(iou={iou:.3f},dist={dist},gap={gap})"
                            ),
                            score=2,
                        )

            # Triad sandwich rules
            if len(frames) >= 3:
                for i in range(1, len(frames) - 1):
                    prev = frames[i - 1]
                    curr = frames[i]
                    nxt = frames[i + 1]

                    gap1 = curr.frame_num - prev.frame_num
                    gap2 = nxt.frame_num - curr.frame_num
                    if gap1 <= 0 or gap2 <= 0:
                        continue
                    if gap1 > max_pair_frame_gap or gap2 > max_pair_frame_gap:
                        continue
                    if prev.dhash64 is None or curr.dhash64 is None or nxt.dhash64 is None:
                        continue

                    d1 = hamming_distance_64(prev.dhash64, curr.dhash64)
                    d2 = hamming_distance_64(curr.dhash64, nxt.dhash64)
                    if d1 > max_hash_dist or d2 > max_hash_dist:
                        continue

                    if prev.has_focus and nxt.has_focus and (not curr.has_focus):
                        iou_pn = max_iou(prev.focus_boxes, nxt.focus_boxes)
                        if iou_pn >= triad_require_iou:
                            add_susp(
                                video_rel,
                                curr.frame_num,
                                reason=(
                                    "between_two_similar_focus_missing_curr"
                                    f"(d1={d1},d2={d2},gap1={gap1},gap2={gap2},iou_pn={iou_pn:.3f})"
                                ),
                                score=4,
                            )
                    if (not prev.has_focus) and (not nxt.has_focus) and curr.has_focus:
                        add_susp(
                            video_rel,
                            curr.frame_num,
                            reason=(
                                "between_two_similar_nonfocus_but_curr_has_focus"
                                f"(d1={d1},d2={d2},gap1={gap1},gap2={gap2})"
                            ),
                            score=4,
                        )

        # Isolated focus positives
        focus_indices = [i for i, r in enumerate(frames) if r.has_focus]
        if focus_indices:
            for idx in focus_indices:
                r = frames[idx]
                prev_idx = next((j for j in reversed(focus_indices) if j < idx), None)
                nxt_idx = next((j for j in focus_indices if j > idx), None)
                prev_gap = r.frame_num - frames[prev_idx].frame_num if prev_idx is not None else None
                nxt_gap = frames[nxt_idx].frame_num - r.frame_num if nxt_idx is not None else None
                nearest = min([g for g in [prev_gap, nxt_gap] if g is not None], default=None)
                if nearest is None or nearest >= isolated_window:
                    add_susp(
                        video_rel,
                        r.frame_num,
                        reason=f"isolated_focus(nearest_gap={nearest})",
                        score=1,
                    )

        # Area outliers within video (robust MAD)
        areas: list[float] = []
        area_by_frame: dict[int, float] = {}
        for r in frames:
            if not r.focus_boxes:
                continue
            a = max(b.area() for b in r.focus_boxes)
            areas.append(a)
            area_by_frame[r.frame_num] = a
        if len(areas) >= 6:
            med = _median(areas)
            if med is not None:
                mad = _mad(areas, med) or 0.0
                if mad > 0:
                    for fn, a in area_by_frame.items():
                        if abs(a - med) > outlier_mad_k * mad:
                            add_susp(
                                video_rel,
                                fn,
                                reason=f"focus_area_outlier(area={a:.6f},med={med:.6f},mad={mad:.6f})",
                                score=1,
                            )

    suspicious_list: list[dict[str, Any]] = []
    for (video_rel_s, frame_num), item in suspicious.items():
        item["reasons"] = sorted(item["reasons"])
        rec = rec_map.get((Path(video_rel_s), frame_num))
        if rec is not None:
            item["txt_path"] = str(rec.txt_path)
            item["jpg_path"] = str(rec.jpg_path)
        suspicious_list.append(item)

    suspicious_list.sort(key=lambda x: (-x["score"], x["video_rel"], x["frame_num"]))

    if min_score > 1:
        suspicious_list = [it for it in suspicious_list if int(it["score"]) >= min_score]

    return {
        "scan": scan_meta,
        "stats": {
            "videos_with_annotations": len(by_video),
            "annotated_frames": len(records),
            "frames_with_focus": frames_with_focus,
            "class_line_counts": dict(sorted(class_line_counts.items(), key=lambda kv: kv[0])),
        },
        "params": {
            "focus_cls": focus_cls,
            "compute_hash": compute_hash,
            "max_pair_frame_gap": max_pair_frame_gap,
            "max_hash_dist": max_hash_dist,
            "neighbor_min_iou": neighbor_min_iou,
            "triad_require_iou": triad_require_iou,
            "isolated_window": isolated_window,
            "outlier_mad_k": outlier_mad_k,
            "min_score": min_score,
            "max_txt": max_txt,
        },
        "results": {
            "suspicious_frames": suspicious_list,
        },
    }


def write_outputs(report: dict[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    susp_csv = output_dir / "suspicious_frames.csv"
    with susp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "video_rel", "frame_num", "reasons", "txt_path", "jpg_path"])
        for it in report["results"]["suspicious_frames"]:
            w.writerow(
                [
                    it["score"],
                    it["video_rel"],
                    it["frame_num"],
                    ";".join(it["reasons"]),
                    it.get("txt_path", ""),
                    it.get("jpg_path", ""),
                ]
            )

    logger.info("Wrote: %s", str(output_dir / "report.json"))
    logger.info("Wrote: %s", str(susp_csv))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find inconsistent internal annotations (focus on one class).")
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("/home/yangxinyu/Test/Data/internalVideos_fireRelated_staticFrames"),
        help="帧图像根目录",
    )
    parser.add_argument(
        "--ann-root",
        type=Path,
        default=Path("/home/yangxinyu/Test/Data/internalVideos_fireRelated_keyFrameAnnotations_v2_static"),
        help="标注根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("boxAnnotation/InconsistentAnnotations/fire/"),
        help="输出目录（CSV/JSON）",
    )
    parser.add_argument("--focus-cls", type=int, default=0, help="只关注该类别（默认: 0）")
    parser.add_argument("--min-score", type=int, default=1, help="只输出score>=min-score的疑似样本（默认: 1）")
    parser.add_argument("--max-txt", type=int, default=0, help="只扫描前N个txt(0表示全量)，用于快速试跑")

    parser.add_argument("--no-hash", action="store_true", help="跳过图像dHash（将禁用相似帧规则）")
    parser.add_argument(
        "--dhash-cache",
        type=Path,
        default=Path("boxAnnotation/InconsistentAnnotations/dhash_cache.csv"),
        help="dHash缓存文件路径（CSV），用于加速重复运行",
    )
    parser.add_argument("--no-dhash-cache", action="store_true", help="禁用dHash缓存读写")

    # Heuristics (use current tested defaults)
    parser.add_argument("--max-pair-frame-gap", type=int, default=60)
    parser.add_argument("--max-hash-dist", type=int, default=8)
    parser.add_argument("--neighbor-min-iou", type=float, default=0.3)
    parser.add_argument("--triad-require-iou", type=float, default=0.6)
    parser.add_argument("--isolated-window", type=int, default=500)
    parser.add_argument("--outlier-mad-k", type=float, default=6.0)

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="dHash并行计算的进程数（0表示使用CPU核心数，1表示单进程）",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dhash_cache: Optional[dict[str, int]] = None
    if (not args.no_hash) and (not args.no_dhash_cache):
        dhash_cache = load_dhash_cache(args.dhash_cache)

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    
    report = analyze_inconsistent_annotations(
        frames_root=args.frames_root,
        ann_root=args.ann_root,
        focus_cls=args.focus_cls,
        compute_hash=(not args.no_hash),
        max_pair_frame_gap=args.max_pair_frame_gap,
        max_hash_dist=args.max_hash_dist,
        neighbor_min_iou=args.neighbor_min_iou,
        triad_require_iou=args.triad_require_iou,
        isolated_window=args.isolated_window,
        outlier_mad_k=args.outlier_mad_k,
        min_score=args.min_score,
        max_txt=args.max_txt,
        dhash_cache=dhash_cache,
        num_workers=num_workers,
    )

    if dhash_cache is not None:
        save_dhash_cache(dhash_cache, args.dhash_cache)

    write_outputs(report, output_dir=args.output_dir)

    stats = report["stats"]
    logger.info(
        "Stats: videos=%d, annotated_frames=%d, frames_with_focus=%d, class_line_counts=%s",
        stats["videos_with_annotations"],
        stats["annotated_frames"],
        stats["frames_with_focus"],
        stats["class_line_counts"],
    )
    logger.info("Suspicious frames: %d", len(report["results"]["suspicious_frames"]))
    for it in report["results"]["suspicious_frames"][:20]:
        logger.info(
            "TOP score=%s %s/frame_%s reasons=%s",
            it["score"],
            it["video_rel"],
            str(it["frame_num"]).zfill(5),
            it["reasons"][0] if it["reasons"] else "",
        )


if __name__ == "__main__":
    main()

