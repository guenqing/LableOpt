from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..models import BBox, BoxSource
from .yolo_utils import pixel_to_yolo, yolo_to_pixel


def boxes_to_backup_payload(gt_boxes: List[BBox], pred_boxes: List[BBox]) -> Dict[str, Any]:
    """Serialize boxes to a JSON-friendly payload for extend backup/restore."""
    def _to_dict(b: BBox) -> Dict[str, Any]:
        src = b.source.value if isinstance(b.source, BoxSource) else str(b.source)
        return {
            "x": float(b.x),
            "y": float(b.y),
            "w": float(b.w),
            "h": float(b.h),
            "class_id": int(b.class_id),
            "source": src,
            "visible": bool(getattr(b, "visible", True)),
            "editable": bool(getattr(b, "editable", True)),
        }

    return {
        "gt": [_to_dict(b) for b in gt_boxes],
        "pred": [_to_dict(b) for b in pred_boxes],
    }


def boxes_from_backup_payload(payload: Dict[str, Any]) -> Tuple[List[BBox], List[BBox]]:
    """Deserialize boxes from backup payload."""
    def _src(s: str) -> BoxSource:
        if str(s).lower() == "pred":
            return BoxSource.PRED
        return BoxSource.GT

    def _from_dict(d: Dict[str, Any]) -> BBox:
        return BBox(
            x=float(d["x"]),
            y=float(d["y"]),
            w=float(d["w"]),
            h=float(d["h"]),
            class_id=int(d["class_id"]),
            source=_src(d.get("source", "gt")),
            visible=bool(d.get("visible", True)),
            editable=bool(d.get("editable", True)),
        )

    gt = [_from_dict(d) for d in (payload.get("gt") or [])]
    pred = [_from_dict(d) for d in (payload.get("pred") or [])]
    return gt, pred


def dumps_backup_payload(payload: Dict[str, Any]) -> str:
    """Dump backup payload to JSON string (stable keys)."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def loads_backup_payload(text: str) -> Dict[str, Any]:
    """Load backup payload from JSON string."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("invalid backup payload")
    return obj


def _replacement_group(class_id: int) -> int:
    """
    Map class_id into a "replacement group" for extend override mode.

    User requirement:
    - 0 or 2 replace 0 and 2 (same object family: TP/FP)
    - 1 or 3 replace 1 and 3 (another family: TP/FP)
    - other class_ids: only replace themselves (unique group)
    """
    cid = int(class_id)
    if cid in (0, 2):
        return 0
    if cid in (1, 3):
        return 1
    return 1000 + cid


def _bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _box_xyxy(box: BBox) -> Tuple[float, float, float, float]:
    x1 = float(box.x)
    y1 = float(box.y)
    x2 = float(box.x + box.w)
    y2 = float(box.y + box.h)
    return x1, y1, x2, y2


def editable_boxes_to_yolo(
    all_boxes: List[BBox],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """
    Convert current editable boxes (pixel) to YOLO-normalized boxes for cross-frame reuse.

    Note:
    - We intentionally copy "editable=True" boxes regardless of their original source (GT/PRED),
      because this represents the annotation the user is actively refining.
    - Output format is a list of dicts: {class_id, cx, cy, w, h}.
    """
    yolo_boxes: List[Dict[str, Any]] = []
    for box in all_boxes:
        if not getattr(box, "editable", True):
            continue
        x1 = float(box.x)
        y1 = float(box.y)
        x2 = float(box.x + box.w)
        y2 = float(box.y + box.h)
        cx, cy, w, h = pixel_to_yolo(x1, y1, x2, y2, img_w, img_h)
        yolo_boxes.append(
            {
                "class_id": int(box.class_id),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
            }
        )
    return yolo_boxes


def yolo_to_gt_boxes(
    yolo_boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> List[BBox]:
    """
    Convert YOLO-normalized boxes to pixel BBox list for the target frame.

    All returned boxes are treated as GT boxes (green) and editable (solid line).
    """
    boxes: List[BBox] = []
    for b in yolo_boxes:
        x1, y1, x2, y2 = yolo_to_pixel(float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"]), img_w, img_h)
        boxes.append(
            BBox(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
                class_id=int(b["class_id"]),
                source=BoxSource.GT,
                visible=True,
                editable=True,
            )
        )
    return boxes


def apply_extend_gt_to_next(
    gt_boxes: List[BBox],
    pred_boxes: List[BBox],
    copied_editable_yolo: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    *,
    prefer_previous_on_overlap: bool = False,
    iou_threshold: float = 0.2,
) -> Tuple[List[BBox], List[BBox], int]:
    """
    Apply "Extend GT to Next" rule on the target frame:
    - First copy previous-frame editable boxes into the target frame (as GT, editable=True)
    - Then handle overlaps in one of two ways:
      - prefer_previous_on_overlap=False (default):
        drop copied boxes that have the SAME class_id and IoU > threshold with ANY existing editable box
        keep the target frame's existing boxes unchanged (prefer current-frame annotations)
      - prefer_previous_on_overlap=True:
        allow "same-group replacement" and prefer previous frame:
        if a copied box overlaps (IoU > threshold) with ANY existing editable box in the SAME replacement group,
        remove those existing boxes and keep the copied box.

    Returns:
        (new_gt_boxes, new_pred_boxes, injected_count_kept)
    """
    injected_all = yolo_to_gt_boxes(copied_editable_yolo, img_w, img_h)

    thr = float(iou_threshold)
    if thr <= 0.0:
        thr = 0.0
    if thr > 1.0:
        thr = 1.0

    # Existing editable boxes in the target frame (regardless of GT/PRED source)
    existing_editable: List[BBox] = [
        b for b in (gt_boxes + pred_boxes) if getattr(b, "editable", True)
    ]

    injected_kept: List[BBox] = []
    if not prefer_previous_on_overlap:
        for inj in injected_all:
            # Only de-dup against same-class editable boxes
            drop = False
            for ex in existing_editable:
                if int(ex.class_id) != int(inj.class_id):
                    continue
                if _bbox_iou_xyxy(_box_xyxy(ex), _box_xyxy(inj)) > thr:
                    drop = True
                    break
            if not drop:
                injected_kept.append(inj)

        # Keep target frame boxes; append new boxes to GT
        new_gt = list(gt_boxes) + injected_kept
        new_pred = list(pred_boxes)
        return new_gt, new_pred, len(injected_kept)

    # prefer_previous_on_overlap=True:
    # We remove overlapping existing editable boxes in the same replacement group,
    # and we keep injected boxes.
    gt_out: List[BBox] = list(gt_boxes)
    pred_out: List[BBox] = list(pred_boxes)

    def _should_remove_existing(existing: BBox, injected: BBox) -> bool:
        if not getattr(existing, "editable", True):
            return False
        if _replacement_group(existing.class_id) != _replacement_group(injected.class_id):
            return False
        return _bbox_iou_xyxy(_box_xyxy(existing), _box_xyxy(injected)) > thr

    for inj in injected_all:
        # Remove overlaps from BOTH gt and pred editable boxes (keep refs untouched)
        gt_out = [b for b in gt_out if not _should_remove_existing(b, inj)]
        pred_out = [b for b in pred_out if not _should_remove_existing(b, inj)]
        injected_kept.append(inj)

    # Append injected boxes to GT
    gt_out = gt_out + injected_kept
    return gt_out, pred_out, len(injected_kept)

    # Keep target frame boxes; append new boxes to GT
    new_gt = list(gt_boxes) + injected_kept
    new_pred = list(pred_boxes)
    return new_gt, new_pred, len(injected_kept)

