import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def yolo_to_pixel(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """YOLO normalized coords -> pixel coords (x1, y1, x2, y2)"""
    box_w = w * img_w
    box_h = h * img_h
    x1 = (cx * img_w) - (box_w / 2)
    y1 = (cy * img_h) - (box_h / 2)
    x2 = x1 + box_w
    y2 = y1 + box_h
    return x1, y1, x2, y2


def pixel_to_yolo(x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Pixel coords (x1, y1, x2, y2) -> YOLO normalized coords"""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def read_yolo_label(label_path: Path, img_w: int, img_h: int,
                    has_confidence: bool = False) -> List[Dict]:
    """
    Read YOLO label file, return pixel coordinate format

    Args:
        label_path: path to label file
        img_w: image width
        img_h: image height
        has_confidence: whether the label file contains confidence scores (pred format)

    Returns:
        List of dicts with 'class_id', 'bbox' [x1,y1,x2,y2], and optionally 'confidence'
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
                box_dict = {
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2]
                }
                if has_confidence and len(parts) >= 6:
                    box_dict['confidence'] = float(parts[5])
                boxes.append(box_dict)
    return boxes


def write_yolo_label(label_path: Path, boxes: List[Dict], img_w: int, img_h: int):
    """Write pixel coordinate format boxes to YOLO label file"""
    with open(label_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            cx, cy, w, h = pixel_to_yolo(x1, y1, x2, y2, img_w, img_h)
            f.write(f"{box['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions (width, height)"""
    with Image.open(image_path) as img:
        return img.size


def collect_image_paths(images_dir: Path) -> List[Path]:
    """
    Collect all image paths from nested directory structure.
    Structure: images_dir/{category}/{video}/frame_xxxxx.jpg

    Returns:
        Sorted list of image paths (relative to images_dir)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []

    for img_path in images_dir.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            # Store relative path
            rel_path = img_path.relative_to(images_dir)
            image_paths.append(rel_path)

    return sorted(image_paths)


def prepare_cleanlab_labels(
    images_dir: Path,
    gt_labels_dir: Path,
    image_rel_paths: List[Path]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Convert YOLO GT labels to Cleanlab format.

    Args:
        images_dir: root directory containing images
        gt_labels_dir: root directory containing GT labels (same structure as images)
        image_rel_paths: list of relative image paths

    Returns:
        labels: cleanlab format label list
        valid_image_paths: list of valid relative image paths (as strings)
    """
    labels = []
    valid_image_paths = []

    for rel_path in image_rel_paths:
        img_path = images_dir / rel_path
        # Corresponding label file (same relative path, different extension)
        label_rel_path = rel_path.with_suffix('.txt')
        label_path = gt_labels_dir / label_rel_path

        # Skip if image doesn't exist (broken symlink, etc.)
        try:
            img_w, img_h = get_image_size(img_path)
        except Exception as e:
            logger.warning(f"Cannot read image {img_path}: {e}, skipping")
            continue

        # Read GT labels
        boxes = read_yolo_label(label_path, img_w, img_h, has_confidence=False)

        if boxes:
            bboxes = np.array([b['bbox'] for b in boxes], dtype=np.float32)
            class_ids = np.array([b['class_id'] for b in boxes], dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_ids = np.array([], dtype=np.int64)

        labels.append({
            'bboxes': bboxes,
            'labels': class_ids,
        })
        valid_image_paths.append(str(rel_path))

    return labels, valid_image_paths


def prepare_cleanlab_predictions(
    images_dir: Path,
    pred_labels_dir: Path,
    image_rel_paths: List[str],
    num_classes: int
) -> List[np.ndarray]:
    """
    Convert YOLO predictions to Cleanlab format.

    Args:
        images_dir: root directory containing images
        pred_labels_dir: root directory containing pred labels (same structure as images)
        image_rel_paths: list of relative image paths (strings)
        num_classes: total number of classes

    Returns:
        predictions: cleanlab format prediction list
    """
    predictions = []

    for rel_path_str in image_rel_paths:
        rel_path = Path(rel_path_str)
        img_path = images_dir / rel_path
        pred_rel_path = rel_path.with_suffix('.txt')
        pred_path = pred_labels_dir / pred_rel_path

        # Get image size
        try:
            img_w, img_h = get_image_size(img_path)
        except Exception as e:
            logger.warning(f"Cannot read image {img_path}: {e}")
            # Return empty predictions for this image
            pred_array = np.array([
                np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)
            ], dtype=object)
            predictions.append(pred_array)
            continue

        # Organize predictions by class
        pred_by_class = [[] for _ in range(num_classes)]

        if pred_path.exists():
            boxes = read_yolo_label(pred_path, img_w, img_h, has_confidence=True)
            for box in boxes:
                class_id = box['class_id']
                conf = box.get('confidence', 1.0)
                x1, y1, x2, y2 = box['bbox']
                if 0 <= class_id < num_classes:
                    pred_by_class[class_id].append([x1, y1, x2, y2, conf])

        # Convert to numpy arrays
        pred_array = np.array([
            np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
            for boxes in pred_by_class
        ], dtype=object)

        predictions.append(pred_array)

    return predictions


def count_classes(
    gt_labels_dir: Path,
    pred_labels_dir: Path,
    image_rel_paths: List[str]
) -> int:
    """
    Count the number of classes from GT and pred labels.

    Returns:
        num_classes: max(class_id) + 1
    """
    all_classes = set()

    for rel_path_str in image_rel_paths:
        rel_path = Path(rel_path_str)

        # Check GT
        gt_path = gt_labels_dir / rel_path.with_suffix('.txt')
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        all_classes.add(int(parts[0]))

        # Check pred
        pred_path = pred_labels_dir / rel_path.with_suffix('.txt')
        if pred_path.exists():
            with open(pred_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        all_classes.add(int(parts[0]))

    return max(all_classes) + 1 if all_classes else 1
