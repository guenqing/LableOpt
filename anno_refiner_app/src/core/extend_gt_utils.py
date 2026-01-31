from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..models import BBox, BoxSource
from .yolo_utils import pixel_to_yolo, yolo_to_pixel


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
) -> Tuple[List[BBox], List[BBox], int]:
    """
    Apply "Extend GT to Next" rule on the target frame:
    - First copy previous-frame editable boxes into the target frame (as GT, editable=True)
    - Then de-duplicate against the target frame's existing editable boxes:
      drop copied boxes that have the SAME class_id and IoU > 0.45 with ANY existing editable box
    - Keep the target frame's existing boxes unchanged (prefer current-frame annotations)

    Returns:
        (new_gt_boxes, new_pred_boxes, injected_count_kept)
    """
    # Existing editable boxes in the target frame (regardless of GT/PRED source)
    existing_editable: List[BBox] = [
        b for b in (gt_boxes + pred_boxes) if getattr(b, "editable", True)
    ]

    injected_all = yolo_to_gt_boxes(copied_editable_yolo, img_w, img_h)

    injected_kept: List[BBox] = []
    for inj in injected_all:
        # Only de-dup against same-class editable boxes
        drop = False
        for ex in existing_editable:
            if int(ex.class_id) != int(inj.class_id):
                continue
            if _bbox_iou_xyxy(_box_xyxy(ex), _box_xyxy(inj)) > 0.45:
                drop = True
                break
        if not drop:
            injected_kept.append(inj)

    # Keep target frame boxes; append new boxes to GT
    new_gt = list(gt_boxes) + injected_kept
    new_pred = list(pred_boxes)
    return new_gt, new_pred, len(injected_kept)

