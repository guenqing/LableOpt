import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

# Threshold for switching to parallel processing
PARALLEL_THRESHOLD = 10000  # Use parallel for > 50k samples


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
    """Get image dimensions (width, height) using Python built-in libraries"""
    # Use struct to parse JPEG header - avoids PIL dependency and DLL issues
    with open(image_path, 'rb') as f:
        # JPEG SOI marker
        f.read(2)
        while True:
            # Find next marker (starts with 0xFF)
            while True:
                marker = f.read(1)
                if marker == b'':
                    raise IOError("Invalid JPEG file")
                if marker == b'\xff':
                    break
            # Read marker type
            marker_type = f.read(1)
            # Read segment length (2 bytes, big-endian)
            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                raise IOError("Invalid JPEG file")
            length = (length_bytes[0] << 8) | length_bytes[1]
            # Skip marker payload (subtract 2 for the length bytes themselves)
            if marker_type in (b'\xc0', b'\xc1', b'\xc2', b'\xc3', b'\xc5', b'\xc6', b'\xc7', b'\xc9', b'\xca', b'\xcb', b'\xcd', b'\xce', b'\xcf'):
                # SOF (Start of Frame) markers contain image dimensions
                # Skip 5 bytes: 1 byte precision, 2 bytes height, 2 bytes width
                f.read(1)  # precision
                height_bytes = f.read(2)
                width_bytes = f.read(2)
                if len(height_bytes) != 2 or len(width_bytes) != 2:
                    raise IOError("Invalid JPEG file")
                height = (height_bytes[0] << 8) | height_bytes[1]
                width = (width_bytes[0] << 8) | width_bytes[1]
                return width, height
            else:
                # Skip other segments
                f.read(length - 2)


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


def collect_label_keys(labels_dir: Path, exclude_tmp: bool = True) -> set[Path]:
    """
    Collect label "keys" from a YOLO labels directory.

    A "key" is the relative path without suffix, e.g.:
      labels_dir/a/b/x.txt -> Path("a/b/x")

    Args:
        labels_dir: labels root directory
        exclude_tmp: whether to exclude *_tmp.txt files

    Returns:
        Set of relative Path keys (without suffix)
    """
    keys: set[Path] = set()
    if not labels_dir.exists():
        return keys

    for p in labels_dir.rglob('*.txt'):
        if not p.is_file():
            continue
        if exclude_tmp and p.name.endswith('_tmp.txt'):
            continue
        rel = p.relative_to(labels_dir)
        keys.add(rel.with_suffix(''))
    return keys


def filter_image_paths_by_label_keys(
    image_rel_paths: List[Path],
    gt_keys: set[Path],
    pred_keys: set[Path],
) -> List[Path]:
    """
    Filter image relative paths by the intersection of GT and Pred label keys.

    Args:
        image_rel_paths: image relative paths (relative to images root)
        gt_keys: keys collected from GT label files (relative paths without suffix)
        pred_keys: keys collected from Pred label files (relative paths without suffix)

    Returns:
        Filtered list of image relative paths that have both GT and Pred labels.
    """
    filtered: List[Path] = []
    for rel_path in image_rel_paths:
        key = rel_path.with_suffix('')
        if key in gt_keys and key in pred_keys:
            filtered.append(rel_path)
    return filtered


def find_image_rel_path_for_key(
    images_dir: Path,
    key: Path,
    image_extensions: Optional[set[str]] = None,
) -> Optional[Path]:
    """
    Resolve an image relative path from a key (relative path without suffix).

    Args:
        images_dir: images root directory
        key: relative path without suffix, e.g. Path("a/b/x")
        image_extensions: allowed extensions (lowercase, include leading dot)

    Returns:
        Relative image path with a concrete extension, or None if not found.
    """
    exts = image_extensions or {'.jpg', '.jpeg', '.png', '.bmp'}
    # deterministic preference for common extensions
    ordered_exts = [e for e in ['.jpg', '.jpeg', '.png', '.bmp'] if e in exts] + sorted(exts - {'.jpg', '.jpeg', '.png', '.bmp'})
    for ext in ordered_exts:
        rel_img = key.with_suffix(ext)
        p = images_dir / rel_img
        if p.is_file():
            return rel_img
    return None


def _process_single_image_gt_worker(args: Tuple[str, str, str]) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Worker function for processing single image GT label (must be top-level for pickling).
    
    Args:
        args: (images_dir_str, gt_labels_dir_str, rel_path_str)
    
    Returns:
        (label_dict, rel_path_str) or None if failed
    """
    images_dir_str, gt_labels_dir_str, rel_path_str = args
    images_dir = Path(images_dir_str)
    gt_labels_dir = Path(gt_labels_dir_str)
    rel_path = Path(rel_path_str)
    
    img_path = images_dir / rel_path
    label_rel_path = rel_path.with_suffix('.txt')
    label_path = gt_labels_dir / label_rel_path

    try:
        img_w, img_h = get_image_size(img_path)
    except Exception as e:
        # Don't log in worker to avoid log spam
        return None

    boxes = read_yolo_label(label_path, img_w, img_h, has_confidence=False)

    if boxes:
        bboxes = np.array([b['bbox'] for b in boxes], dtype=np.float32)
        class_ids = np.array([b['class_id'] for b in boxes], dtype=np.int64)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)
        class_ids = np.array([], dtype=np.int64)

    label_dict = {
        'bboxes': bboxes,
        'labels': class_ids,
    }
    return label_dict, rel_path_str


def prepare_cleanlab_labels(
    images_dir: Path,
    gt_labels_dir: Path,
    image_rel_paths: List[Path]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Convert YOLO GT labels to Cleanlab format.
    Automatically uses parallel processing for large datasets.

    Args:
        images_dir: root directory containing images
        gt_labels_dir: root directory containing GT labels (same structure as images)
        image_rel_paths: list of relative image paths

    Returns:
        labels: cleanlab format label list
        valid_image_paths: list of valid relative image paths (as strings)
    """
    num_samples = len(image_rel_paths)
    
    # Use parallel processing for large datasets
    if num_samples >= PARALLEL_THRESHOLD:
        num_workers = min(multiprocessing.cpu_count(), 32)
        logger.info(f"Using parallel processing for GT labels ({num_samples} samples, {num_workers} workers)")
        
        # Prepare arguments (convert to strings for pickling)
        args_list = [
            (str(images_dir), str(gt_labels_dir), str(rel_path))
            for rel_path in image_rel_paths
        ]
        
        # Process in parallel
        chunksize = max(1, len(args_list) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                _process_single_image_gt_worker,
                args_list,
                chunksize=chunksize
            ))
        
        # Collect valid results
        labels = []
        valid_image_paths = []
        failed_count = 0
        for result in results:
            if result is not None:
                label_dict, rel_path_str = result
                labels.append(label_dict)
                valid_image_paths.append(rel_path_str)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} images out of {num_samples}")
        
        return labels, valid_image_paths
    
    # Use serial processing for small datasets
    else:
        labels = []
        valid_image_paths = []

        for rel_path in image_rel_paths:
            img_path = images_dir / rel_path
            label_rel_path = rel_path.with_suffix('.txt')
            label_path = gt_labels_dir / label_rel_path

            try:
                img_w, img_h = get_image_size(img_path)
            except Exception as e:
                logger.warning(f"Cannot read image {img_path}: {e}, skipping")
                continue

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


def _process_single_image_pred_worker(args: Tuple[str, str, str, int]) -> np.ndarray:
    """
    Worker function for processing single image prediction (must be top-level for pickling).
    
    Args:
        args: (images_dir_str, pred_labels_dir_str, rel_path_str, num_classes)
    
    Returns:
        pred_array: cleanlab format prediction array
    """
    images_dir_str, pred_labels_dir_str, rel_path_str, num_classes = args
    images_dir = Path(images_dir_str)
    pred_labels_dir = Path(pred_labels_dir_str)
    rel_path = Path(rel_path_str)
    
    img_path = images_dir / rel_path
    pred_rel_path = rel_path.with_suffix('.txt')
    pred_path = pred_labels_dir / pred_rel_path

    try:
        img_w, img_h = get_image_size(img_path)
    except Exception:
        return np.array([
            np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)
        ], dtype=object)

    pred_by_class = [[] for _ in range(num_classes)]

    if pred_path.exists():
        boxes = read_yolo_label(pred_path, img_w, img_h, has_confidence=True)
        for box in boxes:
            class_id = box['class_id']
            conf = box.get('confidence', 1.0)
            x1, y1, x2, y2 = box['bbox']
            if 0 <= class_id < num_classes:
                pred_by_class[class_id].append([x1, y1, x2, y2, conf])

    pred_array = np.array([
        np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
        for boxes in pred_by_class
    ], dtype=object)

    return pred_array


def prepare_cleanlab_predictions(
    images_dir: Path,
    pred_labels_dir: Path,
    image_rel_paths: List[str],
    num_classes: int
) -> List[np.ndarray]:
    """
    Convert YOLO predictions to Cleanlab format.
    Automatically uses parallel processing for large datasets.

    Args:
        images_dir: root directory containing images
        pred_labels_dir: root directory containing pred labels (same structure as images)
        image_rel_paths: list of relative image paths (strings)
        num_classes: total number of classes

    Returns:
        predictions: cleanlab format prediction list
    """
    num_samples = len(image_rel_paths)
    
    # Use parallel processing for large datasets
    if num_samples >= PARALLEL_THRESHOLD:
        num_workers = min(multiprocessing.cpu_count(), 32)
        logger.info(f"Using parallel processing for Pred labels ({num_samples} samples, {num_workers} workers)")
        
        # Prepare arguments
        args_list = [
            (str(images_dir), str(pred_labels_dir), rel_path_str, num_classes)
            for rel_path_str in image_rel_paths
        ]
        
        # Process in parallel
        chunksize = max(1, len(args_list) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            predictions = list(executor.map(
                _process_single_image_pred_worker,
                args_list,
                chunksize=chunksize
            ))
        
        return predictions
    
    # Use serial processing for small datasets
    else:
        predictions = []

        for rel_path_str in image_rel_paths:
            rel_path = Path(rel_path_str)
            img_path = images_dir / rel_path
            pred_rel_path = rel_path.with_suffix('.txt')
            pred_path = pred_labels_dir / pred_rel_path

            try:
                img_w, img_h = get_image_size(img_path)
            except Exception as e:
                logger.warning(f"Cannot read image {img_path}: {e}")
                pred_array = np.array([
                    np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)
                ], dtype=object)
                predictions.append(pred_array)
                continue

            pred_by_class = [[] for _ in range(num_classes)]

            if pred_path.exists():
                boxes = read_yolo_label(pred_path, img_w, img_h, has_confidence=True)
                for box in boxes:
                    class_id = box['class_id']
                    conf = box.get('confidence', 1.0)
                    x1, y1, x2, y2 = box['bbox']
                    if 0 <= class_id < num_classes:
                        pred_by_class[class_id].append([x1, y1, x2, y2, conf])

            pred_array = np.array([
                np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
                for boxes in pred_by_class
            ], dtype=object)

            predictions.append(pred_array)

        return predictions


def _count_classes_single_image_worker(args: Tuple[str, str, str]) -> set:
    """
    Worker function for counting classes from single image (must be top-level for pickling).
    
    Args:
        args: (gt_labels_dir_str, pred_labels_dir_str, rel_path_str)
    
    Returns:
        set of class IDs found in this image
    """
    gt_labels_dir_str, pred_labels_dir_str, rel_path_str = args
    gt_labels_dir = Path(gt_labels_dir_str)
    pred_labels_dir = Path(pred_labels_dir_str)
    rel_path = Path(rel_path_str)
    
    classes = set()

    gt_path = gt_labels_dir / rel_path.with_suffix('.txt')
    if gt_path.exists():
        try:
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes.add(int(parts[0]))
        except Exception:
            pass

    pred_path = pred_labels_dir / rel_path.with_suffix('.txt')
    if pred_path.exists():
        try:
            with open(pred_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes.add(int(parts[0]))
        except Exception:
            pass

    return classes


def count_classes(
    gt_labels_dir: Path,
    pred_labels_dir: Path,
    image_rel_paths: List[str]
) -> int:
    """
    Count the number of classes from GT and pred labels.
    Automatically uses parallel processing for large datasets.

    Args:
        gt_labels_dir: root directory containing GT labels
        pred_labels_dir: root directory containing pred labels
        image_rel_paths: list of relative image paths (strings)

    Returns:
        num_classes: max(class_id) + 1
    """
    num_samples = len(image_rel_paths)
    
    # Use parallel processing for large datasets
    if num_samples >= PARALLEL_THRESHOLD:
        num_workers = min(multiprocessing.cpu_count(), 32)
        logger.info(f"Using parallel processing for counting classes ({num_samples} samples, {num_workers} workers)")
        
        # Prepare arguments
        args_list = [
            (str(gt_labels_dir), str(pred_labels_dir), rel_path_str)
            for rel_path_str in image_rel_paths
        ]
        
        # Process in parallel
        chunksize = max(1, len(args_list) // (num_workers * 4))
        all_classes = set()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(
                _count_classes_single_image_worker,
                args_list,
                chunksize=chunksize
            )
            for classes in results:
                all_classes.update(classes)
        
        return max(all_classes) + 1 if all_classes else 1
    
    # Use serial processing for small datasets
    else:
        all_classes = set()

        for rel_path_str in image_rel_paths:
            rel_path = Path(rel_path_str)

            gt_path = gt_labels_dir / rel_path.with_suffix('.txt')
            if gt_path.exists():
                try:
                    with open(gt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                all_classes.add(int(parts[0]))
                except Exception as e:
                    logger.warning(f"Error reading GT label {gt_path}: {e}")

            pred_path = pred_labels_dir / rel_path.with_suffix('.txt')
            if pred_path.exists():
                try:
                    with open(pred_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                all_classes.add(int(parts[0]))
                except Exception as e:
                    logger.warning(f"Error reading pred label {pred_path}: {e}")

        return max(all_classes) + 1 if all_classes else 1
