from pathlib import Path
from datetime import datetime
import shutil
from typing import List
import logging

logger = logging.getLogger(__name__)


def backup_folder(folder_path: str) -> str:
    """
    Backup a folder.

    Returns:
        Backup directory path
    """
    src = Path(folder_path)
    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {folder_path}")

    # Generate backup directory name
    backup_name = f"{src.name}_bk"
    backup_path = src.parent / backup_name

    # If already exists, add timestamp
    if backup_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{src.name}_bk_{timestamp}"
        backup_path = src.parent / backup_name

    shutil.copytree(src, backup_path)
    logger.info(f"Backed up {src} to {backup_path}")
    return str(backup_path)


def save_tmp_annotation(gt_labels_path: str, image_rel_path: str,
                        boxes: List[dict], img_w: int, img_h: int):
    """
    Save temporary annotation file.

    Args:
        gt_labels_path: GT labels root directory
        image_rel_path: relative image path (e.g., category/video/frame_xxxxx.jpg)
        boxes: list of box dicts with 'class_id' and 'bbox'
        img_w: image width
        img_h: image height
    """
    from .yolo_utils import write_yolo_label

    gt_dir = Path(gt_labels_path)
    rel_path = Path(image_rel_path)
    stem = rel_path.stem
    parent = rel_path.parent

    # Create parent directories if needed
    tmp_dir = gt_dir / parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_dir / f"{stem}_tmp.txt"
    write_yolo_label(tmp_path, boxes, img_w, img_h)
    logger.debug(f"Saved tmp annotation to {tmp_path}")


def confirm_changes(gt_labels_path: str, keep_changes: bool):
    """
    Confirm or discard changes.

    Args:
        gt_labels_path: GT labels directory
        keep_changes: True=overwrite original with tmp, False=delete tmp
    """
    gt_dir = Path(gt_labels_path)
    tmp_files = list(gt_dir.rglob("*_tmp.txt"))

    for tmp_file in tmp_files:
        # Infer original filename from xxx_tmp.txt -> xxx.txt
        original_name = tmp_file.name.replace("_tmp.txt", ".txt")
        original_path = tmp_file.parent / original_name

        if keep_changes:
            # Overwrite original file
            shutil.move(str(tmp_file), str(original_path))
            logger.info(f"Confirmed changes: {tmp_file} -> {original_path}")
        else:
            # Delete temporary file
            tmp_file.unlink()
            logger.info(f"Discarded changes: deleted {tmp_file}")


def get_tmp_files(gt_labels_path: str) -> List[str]:
    """Get all temporary file paths (relative to gt_labels_path)"""
    gt_dir = Path(gt_labels_path)
    tmp_files = []
    for f in gt_dir.rglob("*_tmp.txt"):
        tmp_files.append(str(f.relative_to(gt_dir)))
    return tmp_files


def validate_paths(images_path: str, gt_labels_path: str, pred_labels_path: str) -> dict:
    """
    Validate input paths and return statistics.

    Returns:
        dict with keys: 'valid', 'images_count', 'gt_count', 'pred_count', 'errors'
    """
    result = {
        'valid': True,
        'images_count': 0,
        'gt_count': 0,
        'pred_count': 0,
        'errors': []
    }

    images_dir = Path(images_path)
    gt_dir = Path(gt_labels_path)
    pred_dir = Path(pred_labels_path)

    # Check directories exist
    if not images_dir.exists():
        result['valid'] = False
        result['errors'].append(f"Images directory not found: {images_path}")
    if not gt_dir.exists():
        result['valid'] = False
        result['errors'].append(f"GT labels directory not found: {gt_labels_path}")
    if not pred_dir.exists():
        result['valid'] = False
        result['errors'].append(f"Pred labels directory not found: {pred_labels_path}")

    if not result['valid']:
        return result

    # Count files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    result['images_count'] = sum(
        1 for f in images_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in image_extensions
    )
    result['gt_count'] = sum(
        1 for f in gt_dir.rglob('*.txt')
        if f.is_file() and not f.name.endswith('_tmp.txt')
    )
    result['pred_count'] = sum(
        1 for f in pred_dir.rglob('*.txt') if f.is_file()
    )

    if result['images_count'] == 0:
        result['valid'] = False
        result['errors'].append("No images found in images directory")

    return result
