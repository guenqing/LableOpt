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


def save_tmp_annotation(output_path: str, image_rel_path: str,
                        boxes: List[dict], img_w: int, img_h: int):
    """
    Save temporary annotation file to Output Path.

    Args:
        output_path: Output root directory
        image_rel_path: relative image path (e.g., category/video/frame_xxxxx.jpg)
        boxes: list of box dicts with 'class_id' and 'bbox'
        img_w: image width
        img_h: image height
    """
    from .yolo_utils import write_yolo_label

    output_dir = Path(output_path)
    rel_path = Path(image_rel_path)
    stem = rel_path.stem
    parent = rel_path.parent

    # Create parent directories if needed
    tmp_dir = output_dir / parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_dir / f"{stem}_tmp.txt"
    write_yolo_label(tmp_path, boxes, img_w, img_h)
    logger.debug(f"Saved tmp annotation to {tmp_path}")


def confirm_changes(output_path: str, keep_changes: bool):
    """
    Confirm or discard changes.

    Args:
        output_path: Output labels directory
        keep_changes: True=overwrite original with tmp, False=delete tmp
    """
    output_dir = Path(output_path)
    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_path}")
        return
    
    tmp_files = list(output_dir.rglob("*_tmp.txt"))

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


def get_tmp_files(output_path: str) -> List[str]:
    """Get all temporary file paths (relative to output_path)"""
    output_dir = Path(output_path)
    if not output_dir.exists():
        return []
    
    tmp_files = []
    for f in output_dir.rglob("*_tmp.txt"):
        tmp_files.append(str(f.relative_to(output_dir)))
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


def validate_output_path(output_path: str, gt_labels_path: str, pred_labels_path: str) -> tuple:
    """
    Validate output path and check if it conflicts with GT or Pred paths.
    
    Args:
        output_path: Output path to validate
        gt_labels_path: GT labels path
        pred_labels_path: Pred labels path
    
    Returns:
        (status, message) where status is "ok", "warning", or "error"
    """
    if not output_path or not output_path.strip():
        return ("error", "Output Path is required")
    
    try:
        output_resolved = Path(output_path).resolve()
        gt_resolved = Path(gt_labels_path).resolve() if gt_labels_path else None
        pred_resolved = Path(pred_labels_path).resolve() if pred_labels_path else None
        
        if gt_resolved and output_resolved == gt_resolved:
            return ("warning", "Output Path is the same as GT Labels Path")
        if pred_resolved and output_resolved == pred_resolved:
            return ("warning", "Output Path is the same as Pred Labels Path")
        
        return ("ok", None)
    except Exception as e:
        return ("error", f"Invalid output path: {e}")


def ensure_output_structure(output_path: str, image_rel_paths: List[Path]):
    """
    Ensure output path directory structure matches the image structure.
    
    Args:
        output_path: Output labels root directory
        image_rel_paths: List of relative image paths
    """
    output_dir = Path(output_path)
    
    # Create root directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each image path
    created_dirs = set()
    for rel_path in image_rel_paths:
        target_dir = output_dir / rel_path.parent
        if target_dir not in created_dirs:
            target_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(target_dir)
    
    logger.info(f"Ensured output directory structure in {output_path}")


def should_skip_sample(image_rel_path: str, output_path: str, human_verified_path: str = "") -> bool:
    """
    Check if a sample should be skipped (already processed).
    
    Args:
        image_rel_path: Relative image path (e.g., category/video/frame_xxxxx.jpg)
        output_path: Output labels root directory
        human_verified_path: Human verified annotations path (optional, can be empty)
    
    Returns:
        True if sample should be skipped, False otherwise
    """
    rel_path = Path(image_rel_path)
    label_stem = rel_path.stem
    parent = rel_path.parent
    
    # Check Output Path
    output_dir = Path(output_path)
    if output_dir.exists():
        output_label = output_dir / parent / f"{label_stem}.txt"
        output_tmp = output_dir / parent / f"{label_stem}_tmp.txt"
        if output_label.exists() or output_tmp.exists():
            return True
    
    # Check Human Verified Annotation Path (if set)
    if human_verified_path and human_verified_path.strip():
        human_dir = Path(human_verified_path)
        if human_dir.exists():
            human_label = human_dir / parent / f"{label_stem}.txt"
            human_tmp = human_dir / parent / f"{label_stem}_tmp.txt"
            if human_label.exists() or human_tmp.exists():
                return True
    
    return False
