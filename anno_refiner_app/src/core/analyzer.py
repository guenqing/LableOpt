from typing import List, Dict, Callable, Optional, Any
from pathlib import Path
import numpy as np
import logging
import time

from cleanlab.object_detection.rank import (
    get_label_quality_scores,
    compute_overlooked_box_scores,
    compute_swap_box_scores,
    compute_badloc_box_scores,
    _get_valid_inputs_for_compute_scores
)
from cleanlab.internal.constants import ALPHA

from ..models import IssueItem, IssueType
from .yolo_utils import (
    collect_image_paths,
    prepare_cleanlab_labels,
    prepare_cleanlab_predictions,
    count_classes
)

logger = logging.getLogger(__name__)


class CleanlabAnalyzer:
    """Cleanlab object detection analyzer"""

    def __init__(
        self,
        images_path: str,
        pred_labels_path: str,
        gt_labels_path: str,
        output_path: str = "",
        human_verified_path: str = "",
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.images_path = Path(images_path)
        self.pred_labels_path = Path(pred_labels_path)
        self.gt_labels_path = Path(gt_labels_path)
        self.output_path = output_path
        self.human_verified_path = human_verified_path
        self.progress_callback = progress_callback or (lambda msg, pct: None)

        self.labels: List[Dict[str, Any]] = []        # cleanlab format GT
        self.predictions: List[np.ndarray] = []       # cleanlab format predictions
        self.image_paths: List[str] = []              # relative image paths
        self.num_classes: int = 1

    def _report_progress(self, message: str, percentage: float):
        """Report progress"""
        logger.info(f"[{percentage*100:.0f}%] {message}")
        self.progress_callback(message, percentage)

    def prepare_data(self):
        """Prepare data: format conversion"""
        step_start = time.time()
        self._report_progress("Collecting analysis samples...", 0.05)

        # NOTE:
        # For large datasets, scanning all images (Images Path) can be extremely slow
        # and causes the UI connection to drop. We therefore enumerate samples based on
        # the intersection of:
        #   (1) GT label files (exclude *_tmp.txt)
        #   (2) Pred label files
        #   (3) Existing image files (any supported extension)
        # Then we exclude samples already present in Output/HumanVerified paths.
        from .file_manager import should_skip_sample
        from .yolo_utils import find_image_rel_path_for_key

        gt_label_files = [
            p for p in self.gt_labels_path.rglob('*.txt')
            if p.is_file() and not p.name.endswith('_tmp.txt')
        ]
        total_gt = len(gt_label_files)
        if total_gt == 0:
            raise ValueError(f"No GT label files found in {self.gt_labels_path} (excluding *_tmp.txt)")

        filtered_paths: List[Path] = []
        skipped_count = 0
        missing_pred_count = 0
        missing_image_count = 0

        # throttle progress reports (avoid log spam)
        last_report_ts = 0.0
        for i, gt_file in enumerate(gt_label_files, start=1):
            key = gt_file.relative_to(self.gt_labels_path).with_suffix('')  # e.g. a/b/x

            pred_label_path = self.pred_labels_path / key.with_suffix('.txt')
            if not pred_label_path.exists():
                missing_pred_count += 1
            else:
                img_rel_path = find_image_rel_path_for_key(self.images_path, key)
                if img_rel_path is None:
                    missing_image_count += 1
                elif should_skip_sample(str(img_rel_path), self.output_path, self.human_verified_path):
                    skipped_count += 1
                else:
                    filtered_paths.append(img_rel_path)

            now = time.time()
            if (now - last_report_ts) >= 1.0 or i == total_gt:
                last_report_ts = now
                # map loop progress into 5% -> 10%
                pct = 0.05 + (i / total_gt) * 0.05
                self._report_progress(f"Collecting analysis samples... {i}/{total_gt}", pct)

        filtered_paths = sorted(filtered_paths)
        step_time = time.time() - step_start
        logger.info(
            f"Sample collection done: gt={total_gt} "
            f"candidate={len(filtered_paths)} skipped={skipped_count} "
            f"missing_pred={missing_pred_count} missing_img={missing_image_count} "
            f"(耗时: {step_time:.3f}s)"
        )
        
        step_start = time.time()
        self._report_progress(f"Converting GT labels ({len(filtered_paths)} samples)...", 0.1)

        # Convert GT labels to cleanlab format
        self.labels, self.image_paths = prepare_cleanlab_labels(
            self.images_path,
            self.gt_labels_path,
            filtered_paths
        )
        step_time = time.time() - step_start
        logger.info(f"Prepared {len(self.labels)} valid samples (耗时: {step_time:.3f}s)")

        step_start = time.time()
        self._report_progress(f"Counting classes ({len(self.image_paths)} samples)...", 0.2)

        # Count classes
        self.num_classes = count_classes(
            self.gt_labels_path,
            self.pred_labels_path,
            self.image_paths
        )
        step_time = time.time() - step_start
        logger.info(f"Detected {self.num_classes} classes (耗时: {step_time:.3f}s)")

        step_start = time.time()
        self._report_progress(f"Converting Pred labels ({len(self.image_paths)} samples)...", 0.3)

        # Convert predictions to cleanlab format
        self.predictions = prepare_cleanlab_predictions(
            self.images_path,
            self.pred_labels_path,
            self.image_paths,
            self.num_classes
        )
        step_time = time.time() - step_start
        logger.info(f"Pred labels conversion complete (耗时: {step_time:.3f}s)")

        self._report_progress("Data preparation complete", 0.4)

    def analyze(self, top_k: int = 10) -> Dict[IssueType, List[IssueItem]]:
        """
        Execute Cleanlab analysis.

        Returns:
            IssueItem lists grouped by issue type
        """
        # Pre-compute auxiliary inputs to avoid redundant computation
        # This reduces computation time by sharing similarity matrices and other
        # intermediate results across the three score computation functions
        step_start = time.time()
        self._report_progress("Preparing auxiliary inputs...", 0.45)
        
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(
            alpha=ALPHA,
            labels=self.labels,
            predictions=self.predictions
        )
        prep_time = time.time() - step_start
        logger.info(f"Auxiliary inputs prepared (耗时: {prep_time:.3f}s)")

        step_start = time.time()
        self._report_progress("Computing overlooked scores...", 0.5)

        # Compute overlooked scores using auxiliary inputs
        overlooked_scores = compute_overlooked_box_scores(
            auxiliary_inputs=auxiliary_inputs
        )
        step_time = time.time() - step_start
        logger.info(f"Overlooked scores computed (耗时: {step_time:.3f}s)")

        step_start = time.time()
        self._report_progress("Computing swap scores...", 0.6)

        # Compute swap scores using auxiliary inputs
        swap_scores = compute_swap_box_scores(
            auxiliary_inputs=auxiliary_inputs
        )
        step_time = time.time() - step_start
        logger.info(f"Swap scores computed (耗时: {step_time:.3f}s)")

        step_start = time.time()
        self._report_progress("Computing bad location scores...", 0.7)

        # Compute bad location scores using auxiliary inputs
        badloc_scores = compute_badloc_box_scores(
            auxiliary_inputs=auxiliary_inputs
        )
        step_time = time.time() - step_start
        logger.info(f"Bad location scores computed (耗时: {step_time:.3f}s)")

        step_start = time.time()
        self._report_progress("Ranking issues...", 0.8)

        results = {
            IssueType.OVERLOOKED: [],
            IssueType.SWAPPED: [],
            IssueType.BAD_LOCATED: []
        }

        # For overlooked: score per predicted box (low score = more likely overlooked)
        # We want images where model predicted something but GT didn't have it
        # Filter out nan values (nan means the predicted box matches GT perfectly)
        overlooked_items = []
        for i, (img_path, scores) in enumerate(zip(self.image_paths, overlooked_scores)):
            if len(scores) > 0:
                # Filter out nan values
                valid_mask = ~np.isnan(scores)
                if np.any(valid_mask):
                    valid_scores = scores[valid_mask]
                    valid_indices = np.where(valid_mask)[0]
                    min_idx_in_valid = int(np.argmin(valid_scores))
                    min_score = float(valid_scores[min_idx_in_valid])
                    min_idx = int(valid_indices[min_idx_in_valid])
                    overlooked_items.append(IssueItem(
                        image_path=img_path,
                        issue_type=IssueType.OVERLOOKED,
                        score=min_score,
                        box_index=min_idx
                    ))
        overlooked_items.sort(key=lambda x: x.score)
        results[IssueType.OVERLOOKED] = overlooked_items[:top_k]

        # For swapped: score per GT box (low score = more likely wrong class)
        # Filter out nan values
        swapped_items = []
        for i, (img_path, scores) in enumerate(zip(self.image_paths, swap_scores)):
            if len(scores) > 0:
                valid_mask = ~np.isnan(scores)
                if np.any(valid_mask):
                    valid_scores = scores[valid_mask]
                    valid_indices = np.where(valid_mask)[0]
                    min_idx_in_valid = int(np.argmin(valid_scores))
                    min_score = float(valid_scores[min_idx_in_valid])
                    min_idx = int(valid_indices[min_idx_in_valid])
                    swapped_items.append(IssueItem(
                        image_path=img_path,
                        issue_type=IssueType.SWAPPED,
                        score=min_score,
                        box_index=min_idx
                    ))
        swapped_items.sort(key=lambda x: x.score)
        results[IssueType.SWAPPED] = swapped_items[:top_k]

        # For bad_located: score per GT box (low score = more likely bad location)
        # Filter out nan values
        badloc_items = []
        for i, (img_path, scores) in enumerate(zip(self.image_paths, badloc_scores)):
            if len(scores) > 0:
                valid_mask = ~np.isnan(scores)
                if np.any(valid_mask):
                    valid_scores = scores[valid_mask]
                    valid_indices = np.where(valid_mask)[0]
                    min_idx_in_valid = int(np.argmin(valid_scores))
                    min_score = float(valid_scores[min_idx_in_valid])
                    min_idx = int(valid_indices[min_idx_in_valid])
                    badloc_items.append(IssueItem(
                        image_path=img_path,
                        issue_type=IssueType.BAD_LOCATED,
                        score=min_score,
                        box_index=min_idx
                    ))
        badloc_items.sort(key=lambda x: x.score)
        results[IssueType.BAD_LOCATED] = badloc_items[:top_k]
        ranking_time = time.time() - step_start
        logger.info(f"Issues ranked (耗时: {ranking_time:.3f}s)")

        self._report_progress("Analysis complete", 1.0)

        # Log summary
        logger.info(f"Analysis results:")
        logger.info(f"  Overlooked: {len(results[IssueType.OVERLOOKED])} issues")
        logger.info(f"  Swapped: {len(results[IssueType.SWAPPED])} issues")
        logger.info(f"  Bad Located: {len(results[IssueType.BAD_LOCATED])} issues")

        return results

    def get_label_quality_scores(self) -> np.ndarray:
        """
        Get overall label quality scores for all images.

        Returns:
            Array of shape (N,) with scores between 0 and 1.
            Lower scores indicate more likely mislabeled images.
        """
        return get_label_quality_scores(
            labels=self.labels,
            predictions=self.predictions,
            verbose=False
        )
