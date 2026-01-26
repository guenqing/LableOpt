from pathlib import Path
from datetime import datetime
import shutil
from typing import List, Optional
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

    tmp_files = get_tmp_files(output_path)
    confirm_changes_for_tmp_files(output_path, tmp_files, keep_changes=keep_changes)


def get_tmp_files(output_path: str) -> List[str]:
    """Get all temporary file paths (relative to output_path)"""
    output_dir = Path(output_path)
    if not output_dir.exists():
        return []
    
    tmp_files = []
    for f in output_dir.rglob("*_tmp.txt"):
        tmp_files.append(str(f.relative_to(output_dir)))
    return tmp_files


def confirm_changes_for_tmp_files(output_path: str, tmp_files: List[str], keep_changes: bool) -> None:
    """
    Confirm or discard a specific list of tmp files (relative to output_path).

    This avoids re-scanning the whole output directory and reduces UI blocking time.
    """
    output_dir = Path(output_path)
    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_path}")
        return

    start_time = datetime.now()
    processed = 0
    missing = 0

    for rel in tmp_files:
        tmp_file = output_dir / rel
        if not tmp_file.exists():
            missing += 1
            continue
        if not tmp_file.name.endswith('_tmp.txt'):
            continue

        original_name = tmp_file.name.replace("_tmp.txt", ".txt")
        original_path = tmp_file.parent / original_name

        if keep_changes:
            shutil.move(str(tmp_file), str(original_path))
        else:
            tmp_file.unlink()
        processed += 1

    elapsed = (datetime.now() - start_time).total_seconds()
    action = "Confirmed" if keep_changes else "Discarded"
    logger.info(f"{action} changes: processed={processed} missing={missing} (耗时: {elapsed:.3f}s)")


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

    start_time = datetime.now()
    logger.info(f"Validating paths: images={images_path} gt={gt_labels_path} pred={pred_labels_path}")

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

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Path validation done in {elapsed:.3f}s: "
        f"valid={result['valid']} imgs={result['images_count']} gt={result['gt_count']} pred={result['pred_count']}"
    )
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


def collect_annotation_image_paths(
    images_path: str,
    gt_labels_path: str = "",
    pred_labels_path: str = "",
    output_path: str = "",
    human_verified_path: str = "",
) -> List[str]:
    """
    Collect samples for direct annotation (skip cleanlab).

    Rules:
    - Always based on Images Path.
    - If GT Labels Path is provided: intersect with GT label keys (exclude *_tmp.txt).
    - If Pred Labels Path is provided: intersect with Pred label keys.
    - Always exclude samples that already exist in Output Path or Human Verified Path.

    Returns:
        Sorted list of relative image paths (POSIX), relative to Images Path.
    """
    images_dir = Path(images_path)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    if not output_path or not str(output_path).strip():
        raise ValueError("Output path is required for annotation")

    has_gt = bool(gt_labels_path and str(gt_labels_path).strip())
    has_pred = bool(pred_labels_path and str(pred_labels_path).strip())

    gt_dir = Path(gt_labels_path) if has_gt else None
    pred_dir = Path(pred_labels_path) if has_pred else None

    if gt_dir is not None and not gt_dir.exists():
        raise FileNotFoundError(f"GT labels directory not found: {gt_labels_path}")
    if pred_dir is not None and not pred_dir.exists():
        raise FileNotFoundError(f"Pred labels directory not found: {pred_labels_path}")

    from .yolo_utils import collect_image_paths, collect_label_keys, find_image_rel_path_for_key

    # Case A: images only -> enumerate all images
    if (gt_dir is None) and (pred_dir is None):
        rel_paths = collect_image_paths(images_dir)
        kept: List[str] = []
        for rel in rel_paths:
            if should_skip_sample(str(rel), output_path, human_verified_path):
                continue
            kept.append(rel.as_posix())
        return kept

    # Cases B/C/D: enumerate by label keys (avoid scanning all images when possible)
    keys: Optional[set[Path]] = None
    if gt_dir is not None:
        gt_keys = collect_label_keys(gt_dir, exclude_tmp=True)
        keys = set(gt_keys) if keys is None else (keys & gt_keys)
    if pred_dir is not None:
        pred_keys = collect_label_keys(pred_dir, exclude_tmp=False)
        keys = set(pred_keys) if keys is None else (keys & pred_keys)

    if keys is None:
        return []

    kept: List[str] = []
    for key in sorted(keys):
        img_rel = find_image_rel_path_for_key(images_dir, key)
        if img_rel is None:
            continue
        if should_skip_sample(str(img_rel), output_path, human_verified_path):
            continue
        kept.append(img_rel.as_posix())

    return kept


def estimate_pending_analysis_samples(
    images_path: str,
    gt_labels_path: str,
    pred_labels_path: str,
    output_path: str,
    human_verified_path: str = "",
) -> dict:
    """
    Estimate pending samples for Cleanlab analysis.

    Pending set definition:
      Images ∩ GT ∩ Pred - (Output ∪ HumanVerified)

    Notes:
    - GT tmp files (*_tmp.txt) are excluded from enumeration.
    - Output/HumanVerified treat either {stem}.txt or {stem}_tmp.txt as "processed".
    - This function is intended for dashboard display and should be cheaper than scanning all images.
      It enumerates GT label files and validates Pred/Image existence per key.
    """
    result = {
        'valid': True,
        'pending': 0,
        'total_gt': 0,
        'missing_pred': 0,
        'missing_img': 0,
        'skipped': 0,
        'errors': [],
    }

    images_dir = Path(images_path) if images_path else None
    gt_dir = Path(gt_labels_path) if gt_labels_path else None
    pred_dir = Path(pred_labels_path) if pred_labels_path else None

    if not images_dir or not images_dir.exists():
        result['valid'] = False
        result['errors'].append(f"Images directory not found: {images_path}")
    if not gt_dir or not gt_dir.exists():
        result['valid'] = False
        result['errors'].append(f"GT labels directory not found: {gt_labels_path}")
    if not pred_dir or not pred_dir.exists():
        result['valid'] = False
        result['errors'].append(f"Pred labels directory not found: {pred_labels_path}")
    if not output_path or not str(output_path).strip():
        result['valid'] = False
        result['errors'].append("Output path is required")

    if not result['valid']:
        return result

    from .yolo_utils import find_image_rel_path_for_key

    for gt_file in gt_dir.rglob('*.txt'):
        if not gt_file.is_file():
            continue
        if gt_file.name.endswith('_tmp.txt'):
            continue

        result['total_gt'] += 1
        key = gt_file.relative_to(gt_dir).with_suffix('')

        pred_label_path = pred_dir / key.with_suffix('.txt')
        if not pred_label_path.exists():
            result['missing_pred'] += 1
            continue

        img_rel_path = find_image_rel_path_for_key(images_dir, key)
        if img_rel_path is None:
            result['missing_img'] += 1
            continue

        if should_skip_sample(str(img_rel_path), output_path, human_verified_path):
            result['skipped'] += 1
            continue

        result['pending'] += 1

    return result


def _count_image_files(images_dir: Path) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sum(
        1
        for p in images_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in image_extensions
    )


def _collect_processed_label_keys(labels_dir: Path) -> set[Path]:
    """
    Collect processed label keys from a labels directory.

    A processed key is the relative path without suffix, normalized so that:
      a/b/x.txt     -> a/b/x
      a/b/x_tmp.txt -> a/b/x
    """
    keys: set[Path] = set()
    if not labels_dir.exists():
        return keys

    for p in labels_dir.rglob('*.txt'):
        if not p.is_file():
            continue
        rel = p.relative_to(labels_dir)
        name = rel.name
        if name.endswith('_tmp.txt'):
            base = name[:-len('_tmp.txt')]
        else:
            base = rel.stem
        keys.add(rel.with_name(base).with_suffix(''))

    return keys


def _collect_processed_label_keys_txt_only(labels_dir: Path) -> set[Path]:
    """Collect processed keys from .txt only (exclude *_tmp.txt)."""
    keys: set[Path] = set()
    if not labels_dir.exists():
        return keys

    for p in labels_dir.rglob('*.txt'):
        if not p.is_file():
            continue
        if p.name.endswith('_tmp.txt'):
            continue
        rel = p.relative_to(labels_dir)
        keys.add(rel.with_suffix(''))

    return keys


def parse_data_for_dashboard(
    images_path: str,
    gt_labels_path: str,
    pred_labels_path: str,
    output_path: str,
    human_verified_path: str,
) -> dict:
    """
    Parse dashboard data on demand.

    Returns:
        dict with keys:
          - mode: "images_full" or "labels"
          - images_exists, images_count
          - gt_valid, gt_missing_img
          - pred_valid, pred_missing_img
          - output_valid, output_missing_img
          - human_valid, human_missing_img
          - pending (int or None)
          - errors: list[str]
    """
    result = {
        "mode": "",
        "images_exists": False,
        "images_count": None,
        "gt_valid": 0,
        "gt_missing_img": 0,
        "pred_valid": 0,
        "pred_missing_img": 0,
        "output_valid": 0,
        "output_missing_img": 0,
        "human_valid": 0,
        "human_missing_img": 0,
        "pending": None,
        "errors": [],
    }

    images_dir = Path(images_path) if images_path and str(images_path).strip() else None
    gt_dir = Path(gt_labels_path) if gt_labels_path and str(gt_labels_path).strip() else None
    pred_dir = Path(pred_labels_path) if pred_labels_path and str(pred_labels_path).strip() else None
    output_dir = Path(output_path) if output_path and str(output_path).strip() else None
    human_dir = Path(human_verified_path) if human_verified_path and str(human_verified_path).strip() else None

    if images_dir is not None and images_dir.exists():
        result["images_exists"] = True
    elif images_dir is not None:
        result["errors"].append(f"Images directory not found: {images_path}")

    has_gt = gt_dir is not None
    has_pred = pred_dir is not None
    labels_mode = has_gt or has_pred

    from .yolo_utils import collect_label_keys, find_image_rel_path_for_key

    image_cache: dict[Path, Optional[Path]] = {}

    def _find_image_for_key(key: Path) -> Optional[Path]:
        if key in image_cache:
            return image_cache[key]
        if images_dir is None or not result["images_exists"]:
            image_cache[key] = None
            return None
        rel = find_image_rel_path_for_key(images_dir, key)
        image_cache[key] = rel
        return rel

    if not labels_mode:
        result["mode"] = "images_full"
        if result["images_exists"]:
            result["images_count"] = _count_image_files(images_dir)
            result["pending"] = int(result["images_count"])
        return result

    result["mode"] = "labels"

    gt_keys: Optional[set[Path]] = None
    pred_keys: Optional[set[Path]] = None

    if gt_dir is not None:
        if gt_dir.exists():
            gt_keys = collect_label_keys(gt_dir, exclude_tmp=True)
            for key in gt_keys:
                if _find_image_for_key(key) is None:
                    result["gt_missing_img"] += 1
                else:
                    result["gt_valid"] += 1
        else:
            result["errors"].append(f"GT labels directory not found: {gt_labels_path}")

    if pred_dir is not None:
        if pred_dir.exists():
            pred_keys = collect_label_keys(pred_dir, exclude_tmp=False)
            for key in pred_keys:
                if _find_image_for_key(key) is None:
                    result["pred_missing_img"] += 1
                else:
                    result["pred_valid"] += 1
        else:
            result["errors"].append(f"Pred labels directory not found: {pred_labels_path}")

    output_keys: set[Path] = set()
    if output_dir is not None:
        if output_dir.exists():
            output_keys = _collect_processed_label_keys(output_dir)
            for key in output_keys:
                if _find_image_for_key(key) is None:
                    result["output_missing_img"] += 1
                else:
                    result["output_valid"] += 1
        else:
            result["errors"].append(f"Output directory not found: {output_path}")

    human_keys: set[Path] = set()
    if human_dir is not None:
        if human_dir.exists():
            human_keys = _collect_processed_label_keys_txt_only(human_dir)
            for key in human_keys:
                if _find_image_for_key(key) is None:
                    result["human_missing_img"] += 1
                else:
                    result["human_valid"] += 1
        else:
            result["errors"].append(f"Human verified directory not found: {human_verified_path}")

    if not result["images_exists"]:
        return result

    processed_keys = output_keys | human_keys

    candidate_keys: Optional[set[Path]] = None
    if gt_keys is not None:
        candidate_keys = set(gt_keys) if candidate_keys is None else (candidate_keys & gt_keys)
    if pred_keys is not None:
        candidate_keys = set(pred_keys) if candidate_keys is None else (candidate_keys & pred_keys)

    if not candidate_keys:
        result["pending"] = 0
        return result

    pending = 0
    for key in candidate_keys:
        if _find_image_for_key(key) is None:
            continue
        if key in processed_keys:
            continue
        pending += 1

    result["pending"] = pending
    return result


def estimate_dashboard_counts_and_pending(
    images_path: str = "",
    gt_labels_path: str = "",
    pred_labels_path: str = "",
    output_path: str = "",
    human_verified_path: str = "",
    known_images_count: Optional[int] = None,
    include_images_count: bool = True,
) -> dict:
    """
    Estimate per-path counts and pending samples (for direct annotation queue).

    Pending definition (generalized by provided label paths):
      Images ∩ (GT if set) ∩ (Pred if set) - processed(Output/HumanVerified)

    Returns:
        dict with keys:
          - images_exists, images_count
          - gt_exists, gt_count
          - pred_exists, pred_count
          - output_set, output_exists, output_processed
          - human_set, human_exists, human_processed
          - pending (int or None), pending_mode, pending_skipped, pending_missing_img
          - errors: list[str]
    """
    result = {
        'images_exists': False,
        'images_count': None,
        'gt_exists': False,
        'gt_count': None,
        'pred_exists': False,
        'pred_count': None,
        'output_set': bool(output_path and str(output_path).strip()),
        'output_exists': False,
        'output_processed': None,
        'human_set': bool(human_verified_path and str(human_verified_path).strip()),
        'human_exists': False,
        'human_processed': None,
        'pending': None,
        'pending_mode': '',
        'pending_skipped': 0,
        'pending_missing_img': 0,
        'errors': [],
    }

    images_dir = Path(images_path) if images_path and str(images_path).strip() else None
    gt_dir = Path(gt_labels_path) if gt_labels_path and str(gt_labels_path).strip() else None
    pred_dir = Path(pred_labels_path) if pred_labels_path and str(pred_labels_path).strip() else None
    output_dir = Path(output_path) if output_path and str(output_path).strip() else None
    human_dir = Path(human_verified_path) if human_verified_path and str(human_verified_path).strip() else None

    gt_keys: Optional[set[Path]] = None
    pred_keys: Optional[set[Path]] = None

    if gt_dir is not None:
        if gt_dir.exists():
            result['gt_exists'] = True
            from .yolo_utils import collect_label_keys
            gt_keys = collect_label_keys(gt_dir, exclude_tmp=True)
            result['gt_count'] = len(gt_keys)
        else:
            result['errors'].append(f"GT labels directory not found: {gt_labels_path}")

    if pred_dir is not None:
        if pred_dir.exists():
            result['pred_exists'] = True
            from .yolo_utils import collect_label_keys
            pred_keys = collect_label_keys(pred_dir, exclude_tmp=False)
            result['pred_count'] = len(pred_keys)
        else:
            result['errors'].append(f"Pred labels directory not found: {pred_labels_path}")

    output_keys: set[Path] = set()
    if output_dir is not None:
        result['output_exists'] = output_dir.exists()
        if output_dir.exists():
            output_keys = _collect_processed_label_keys(output_dir)
            result['output_processed'] = len(output_keys)
        else:
            result['output_processed'] = 0

    human_keys: set[Path] = set()
    if human_dir is not None:
        result['human_exists'] = human_dir.exists()
        if human_dir.exists():
            human_keys = _collect_processed_label_keys(human_dir)
            result['human_processed'] = len(human_keys)
        else:
            result['human_processed'] = 0

    if images_dir is not None and images_dir.exists():
        result['images_exists'] = True
    elif images_dir is not None:
        result['errors'].append(f"Images directory not found: {images_path}")

    # Pending is meaningful only when Images exists and Output Path is set (so we can exclude processed).
    if not result['images_exists']:
        return result
    if output_dir is None:
        # still provide images_count if requested (prefer cached value)
        if include_images_count:
            if known_images_count is not None:
                result['images_count'] = int(known_images_count)
            else:
                result['images_count'] = _count_image_files(images_dir)
        else:
            result['images_count'] = int(known_images_count) if known_images_count is not None else None
        return result

    processed_keys = output_keys | human_keys

    from .yolo_utils import find_image_rel_path_for_key

    # Determine pending mode and enumerate candidates
    if gt_dir is None and pred_dir is None:
        # Images-only: use image scan result and subtract processed
        result['pending_mode'] = 'images'
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        pending = 0
        skipped = 0
        total_imgs = 0
        for p in images_dir.rglob('*'):
            if not (p.is_file() and p.suffix.lower() in image_extensions):
                continue
            total_imgs += 1
            rel_img = p.relative_to(images_dir)
            key = rel_img.with_suffix('')
            if key in processed_keys:
                skipped += 1
                continue
            pending += 1
        result['images_count'] = total_imgs
        result['pending'] = pending
        result['pending_skipped'] = skipped
        return result

    # In label-key modes, count images separately (do not enumerate all images for pending).
    if include_images_count:
        if known_images_count is not None:
            result['images_count'] = int(known_images_count)
        else:
            result['images_count'] = _count_image_files(images_dir)
    else:
        result['images_count'] = int(known_images_count) if known_images_count is not None else None

    keys: Optional[set[Path]] = None
    if gt_keys is not None:
        keys = set(gt_keys) if keys is None else (keys & gt_keys)
        result['pending_mode'] = 'images_gt'
    if pred_keys is not None:
        keys = set(pred_keys) if keys is None else (keys & pred_keys)
        result['pending_mode'] = 'images_pred' if result['pending_mode'] == '' else 'images_gt_pred'

    if not keys:
        result['pending'] = 0
        return result

    pending = 0
    skipped = 0
    missing_img = 0
    for key in keys:
        img_rel = find_image_rel_path_for_key(images_dir, key)
        if img_rel is None:
            missing_img += 1
            continue
        if key in processed_keys:
            skipped += 1
            continue
        pending += 1

    result['pending'] = pending
    result['pending_skipped'] = skipped
    result['pending_missing_img'] = missing_img
    return result


def count_images_in_dir(images_path: str) -> int:
    """Count image files under Images Path."""
    images_dir = Path(images_path)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    return _count_image_files(images_dir)
