from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from .yolo_utils import (
    get_image_size,
    collect_image_paths,
    find_image_rel_path_for_key,
    read_yolo_label,
    write_yolo_label,
)

logger = logging.getLogger(__name__)

SUPPORTED_LABEL_EXTENSIONS = ('.txt', '.xml')
PARALLEL_THRESHOLD = 10000


def build_class_name_to_id(class_mapping: Optional[Any]) -> Optional[Dict[str, int]]:
    if class_mapping is None:
        return None

    id_to_name = getattr(class_mapping, 'id_to_name', None)
    if isinstance(id_to_name, dict):
        return {str(name): int(class_id) for class_id, name in id_to_name.items()}

    if isinstance(class_mapping, dict):
        return {str(name): int(class_id) for name, class_id in class_mapping.items()}

    return None


def build_class_id_to_name(class_mapping: Optional[Any]) -> Optional[Dict[int, str]]:
    if class_mapping is None:
        return None

    id_to_name = getattr(class_mapping, 'id_to_name', None)
    if isinstance(id_to_name, dict):
        return {int(class_id): str(name) for class_id, name in id_to_name.items()}

    if isinstance(class_mapping, dict):
        return {int(class_id): str(name) for class_id, name in class_mapping.items()}

    return None


def is_tmp_label_file(path: Path) -> bool:
    return any(path.name.endswith(f'_tmp{ext}') for ext in SUPPORTED_LABEL_EXTENSIONS)


def strip_tmp_suffix(path: Path) -> Path:
    for ext in SUPPORTED_LABEL_EXTENSIONS:
        suffix = f'_tmp{ext}'
        if path.name.endswith(suffix):
            base_name = path.name[:-len(suffix)]
            return path.with_name(base_name)
    return path


def iter_label_files(labels_dir: Path, exclude_tmp: bool = True) -> List[Path]:
    files: List[Path] = []
    if not labels_dir.exists():
        return files

    for ext in SUPPORTED_LABEL_EXTENSIONS:
        for path in labels_dir.rglob(f'*{ext}'):
            if not path.is_file():
                continue
            if exclude_tmp and is_tmp_label_file(path):
                continue
            files.append(path)

    return sorted(files)


def collect_label_keys(labels_dir: Path, exclude_tmp: bool = True) -> set[Path]:
    keys: set[Path] = set()
    for path in iter_label_files(labels_dir, exclude_tmp=exclude_tmp):
        rel = path.relative_to(labels_dir)
        if is_tmp_label_file(path):
            rel = strip_tmp_suffix(rel)
        keys.add(rel.with_suffix(''))
    return keys


def resolve_label_path(labels_dir: Path, key: Path, include_tmp: bool = False, preferred_ext: Optional[str] = None) -> Optional[Path]:
    if not labels_dir or not labels_dir.exists():
        return None

    extensions = list(SUPPORTED_LABEL_EXTENSIONS)
    if preferred_ext in SUPPORTED_LABEL_EXTENSIONS:
        extensions.remove(preferred_ext)
        extensions.insert(0, preferred_ext)

    for ext in extensions:
        candidate = labels_dir / key.parent / f'{key.stem}_tmp{ext}' if include_tmp else labels_dir / key.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


def get_label_extension(labels_dir: Path, key: Path, default_ext: str = '.txt') -> str:
    resolved = resolve_label_path(labels_dir, key)
    if resolved is not None:
        return resolved.suffix.lower()
    return default_ext


def get_output_label_path(output_dir: Path, image_rel_path: Path, label_ext: str, tmp: bool = False) -> Path:
    stem = image_rel_path.stem
    filename = f'{stem}_tmp{label_ext}' if tmp else f'{stem}{label_ext}'
    return output_dir / image_rel_path.parent / filename


def _class_name_to_id(class_name: str, class_name_to_id: Optional[Dict[str, int]]) -> int:
    name = (class_name or '').strip()
    if name == '':
        return 0

    if class_name_to_id and name in class_name_to_id:
        return int(class_name_to_id[name])

    try:
        return int(name)
    except ValueError as exc:
        raise ValueError(
            f"XML class name '{name}' is not numeric and no classes mapping was provided"
        ) from exc


def read_xml_label(
    label_path: Path,
    img_w: int,
    img_h: int,
    has_confidence: bool = False,
    class_name_to_id: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    if not label_path.exists():
        return boxes

    tree = ET.parse(label_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        class_id = _class_name_to_id(name, class_name_to_id)
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue

        try:
            x1 = float(bndbox.findtext('xmin', '0'))
            y1 = float(bndbox.findtext('ymin', '0'))
            x2 = float(bndbox.findtext('xmax', '0'))
            y2 = float(bndbox.findtext('ymax', '0'))
        except ValueError:
            logger.warning(f'Invalid XML box coordinates in {label_path}')
            continue

        x1 = max(0.0, min(x1, float(img_w)))
        y1 = max(0.0, min(y1, float(img_h)))
        x2 = max(0.0, min(x2, float(img_w)))
        y2 = max(0.0, min(y2, float(img_h)))
        if x2 <= x1 or y2 <= y1:
            continue

        box: Dict[str, Any] = {
            'class_id': class_id,
            'bbox': [x1, y1, x2, y2],
        }
        if has_confidence:
            conf_text = obj.findtext('confidence') or obj.findtext('score') or obj.findtext('probability')
            if conf_text is not None:
                try:
                    box['confidence'] = float(conf_text)
                except ValueError:
                    box['confidence'] = 1.0
            else:
                box['confidence'] = 1.0
        boxes.append(box)

    return boxes


def write_xml_label(
    label_path: Path,
    boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    class_id_to_name: Optional[Dict[int, str]] = None,
    image_filename: Optional[str] = None,
) -> None:
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = image_filename or ''

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(int(img_w))
    ET.SubElement(size, 'height').text = str(int(img_h))
    ET.SubElement(size, 'depth').text = '3'

    for box in boxes:
        obj = ET.SubElement(annotation, 'object')
        class_id = int(box['class_id'])
        name = class_id_to_name.get(class_id, str(class_id)) if class_id_to_name else str(class_id)
        ET.SubElement(obj, 'name').text = name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        if 'confidence' in box:
            ET.SubElement(obj, 'confidence').text = f"{float(box['confidence']):.6f}"

        x1, y1, x2, y2 = box['bbox']
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(round(x1)))
        ET.SubElement(bndbox, 'ymin').text = str(int(round(y1)))
        ET.SubElement(bndbox, 'xmax').text = str(int(round(x2)))
        ET.SubElement(bndbox, 'ymax').text = str(int(round(y2)))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(annotation)
    tree.write(label_path, encoding='utf-8', xml_declaration=True)


def read_label_file(
    label_path: Optional[Path],
    img_w: int,
    img_h: int,
    has_confidence: bool = False,
    class_name_to_id: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    if label_path is None or not label_path.exists():
        return []

    suffix = label_path.suffix.lower()
    if suffix == '.xml':
        return read_xml_label(
            label_path,
            img_w,
            img_h,
            has_confidence=has_confidence,
            class_name_to_id=class_name_to_id,
        )

    return read_yolo_label(label_path, img_w, img_h, has_confidence=has_confidence)


def write_label_file(
    label_path: Path,
    boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    class_id_to_name: Optional[Dict[int, str]] = None,
    image_filename: Optional[str] = None,
) -> None:
    suffix = label_path.suffix.lower()
    if suffix == '.xml':
        write_xml_label(
            label_path,
            boxes,
            img_w,
            img_h,
            class_id_to_name=class_id_to_name,
            image_filename=image_filename,
        )
    else:
        write_yolo_label(label_path, boxes, img_w, img_h)


def _process_single_image_gt_worker(args: Tuple[str, str, str, Optional[Dict[str, int]]]) -> Optional[Tuple[Dict[str, Any], str]]:
    images_dir_str, gt_labels_dir_str, rel_path_str, class_name_to_id = args
    images_dir = Path(images_dir_str)
    gt_labels_dir = Path(gt_labels_dir_str)
    rel_path = Path(rel_path_str)
    key = rel_path.with_suffix('')

    img_path = images_dir / rel_path
    label_path = resolve_label_path(gt_labels_dir, key)

    try:
        img_w, img_h = get_image_size(img_path)
    except Exception:
        return None

    try:
        boxes = read_label_file(label_path, img_w, img_h, has_confidence=False, class_name_to_id=class_name_to_id)
    except Exception:
        return None

    if boxes:
        bboxes = np.array([b['bbox'] for b in boxes], dtype=np.float32)
        class_ids = np.array([b['class_id'] for b in boxes], dtype=np.int64)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)
        class_ids = np.array([], dtype=np.int64)

    return {'bboxes': bboxes, 'labels': class_ids}, rel_path_str


def prepare_cleanlab_labels(
    images_dir: Path,
    gt_labels_dir: Path,
    image_rel_paths: List[Path],
    class_name_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    num_samples = len(image_rel_paths)

    if num_samples >= PARALLEL_THRESHOLD:
        num_workers = min(multiprocessing.cpu_count(), 32)
        args_list = [
            (str(images_dir), str(gt_labels_dir), str(rel_path), class_name_to_id)
            for rel_path in image_rel_paths
        ]
        chunksize = max(1, len(args_list) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_single_image_gt_worker, args_list, chunksize=chunksize))

        labels: List[Dict[str, Any]] = []
        valid_image_paths: List[str] = []
        for result in results:
            if result is None:
                continue
            label_dict, rel_path_str = result
            labels.append(label_dict)
            valid_image_paths.append(rel_path_str)
        return labels, valid_image_paths

    labels: List[Dict[str, Any]] = []
    valid_image_paths: List[str] = []
    for rel_path in image_rel_paths:
        img_path = images_dir / rel_path
        label_path = resolve_label_path(gt_labels_dir, rel_path.with_suffix(''))
        try:
            img_w, img_h = get_image_size(img_path)
            boxes = read_label_file(label_path, img_w, img_h, has_confidence=False, class_name_to_id=class_name_to_id)
        except Exception as exc:
            logger.warning(f'Cannot process GT label for {img_path}: {exc}')
            continue

        if boxes:
            bboxes = np.array([b['bbox'] for b in boxes], dtype=np.float32)
            class_ids = np.array([b['class_id'] for b in boxes], dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_ids = np.array([], dtype=np.int64)

        labels.append({'bboxes': bboxes, 'labels': class_ids})
        valid_image_paths.append(str(rel_path))

    return labels, valid_image_paths


def _process_single_image_pred_worker(args: Tuple[str, str, str, int, Optional[Dict[str, int]]]) -> np.ndarray:
    images_dir_str, pred_labels_dir_str, rel_path_str, num_classes, class_name_to_id = args
    images_dir = Path(images_dir_str)
    pred_labels_dir = Path(pred_labels_dir_str)
    rel_path = Path(rel_path_str)
    key = rel_path.with_suffix('')

    img_path = images_dir / rel_path
    pred_path = resolve_label_path(pred_labels_dir, key)

    try:
        img_w, img_h = get_image_size(img_path)
    except Exception:
        return np.array([np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)], dtype=object)

    pred_by_class = [[] for _ in range(num_classes)]
    if pred_path and pred_path.exists():
        try:
            boxes = read_label_file(pred_path, img_w, img_h, has_confidence=True, class_name_to_id=class_name_to_id)
        except Exception:
            boxes = []
        for box in boxes:
            class_id = int(box['class_id'])
            conf = float(box.get('confidence', 1.0))
            x1, y1, x2, y2 = box['bbox']
            if 0 <= class_id < num_classes:
                pred_by_class[class_id].append([x1, y1, x2, y2, conf])

    return np.array([
        np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
        for boxes in pred_by_class
    ], dtype=object)


def prepare_cleanlab_predictions(
    images_dir: Path,
    pred_labels_dir: Path,
    image_rel_paths: List[str],
    num_classes: int,
    class_name_to_id: Optional[Dict[str, int]] = None,
) -> List[np.ndarray]:
    num_samples = len(image_rel_paths)
    if num_samples >= PARALLEL_THRESHOLD:
        num_workers = min(multiprocessing.cpu_count(), 32)
        args_list = [
            (str(images_dir), str(pred_labels_dir), rel_path_str, num_classes, class_name_to_id)
            for rel_path_str in image_rel_paths
        ]
        chunksize = max(1, len(args_list) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(_process_single_image_pred_worker, args_list, chunksize=chunksize))

    predictions: List[np.ndarray] = []
    for rel_path_str in image_rel_paths:
        predictions.append(
            _process_single_image_pred_worker(
                (str(images_dir), str(pred_labels_dir), rel_path_str, num_classes, class_name_to_id)
            )
        )
    return predictions


def _count_classes_single_image_worker(args: Tuple[str, str, str, Optional[Dict[str, int]]]) -> set[int]:
    gt_labels_dir_str, pred_labels_dir_str, rel_path_str, class_name_to_id = args
    gt_labels_dir = Path(gt_labels_dir_str)
    pred_labels_dir = Path(pred_labels_dir_str)
    rel_path = Path(rel_path_str)
    key = rel_path.with_suffix('')
    classes: set[int] = set()

    for labels_dir in (gt_labels_dir, pred_labels_dir):
        label_path = resolve_label_path(labels_dir, key)
        if label_path is None:
            continue
        try:
            boxes = read_label_file(label_path, 1, 1, has_confidence=True, class_name_to_id=class_name_to_id)
        except Exception:
            continue
        for box in boxes:
            classes.add(int(box['class_id']))

    return classes


def count_classes(
    gt_labels_dir: Path,
    pred_labels_dir: Path,
    image_rel_paths: List[str],
    class_name_to_id: Optional[Dict[str, int]] = None,
) -> int:
    num_samples = len(image_rel_paths)
    all_classes: set[int] = set()

    if num_samples >= PARALLEL_THRESHOLD:
        num_workers = min(multiprocessing.cpu_count(), 32)
        args_list = [
            (str(gt_labels_dir), str(pred_labels_dir), rel_path_str, class_name_to_id)
            for rel_path_str in image_rel_paths
        ]
        chunksize = max(1, len(args_list) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for classes in executor.map(_count_classes_single_image_worker, args_list, chunksize=chunksize):
                all_classes.update(classes)
        return max(all_classes) + 1 if all_classes else 1

    for rel_path_str in image_rel_paths:
        all_classes.update(
            _count_classes_single_image_worker(
                (str(gt_labels_dir), str(pred_labels_dir), rel_path_str, class_name_to_id)
            )
        )

    return max(all_classes) + 1 if all_classes else 1
