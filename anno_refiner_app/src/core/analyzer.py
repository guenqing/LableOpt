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
        self._report_progress("Collecting image paths...", 0.05)

        # Collect all image paths from nested structure
        all_image_rel_paths = collect_image_paths(self.images_path)
        step_time = time.time() - step_start
        logger.info(f"Found {len(all_image_rel_paths)} images in {self.images_path} (耗时: {step_time:.3f}s)")

        # Filter out already processed samples
        from .file_manager import should_skip_sample
        filtered_paths = []
        skipped_count = 0
        for rel_path in all_image_rel_paths:
            if should_skip_sample(str(rel_path), self.output_path, self.human_verified_path):
                skipped_count += 1
            else:
                filtered_paths.append(rel_path)
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already processed samples, {len(filtered_paths)} remaining")
        
        step_start = time.time()
        self._report_progress("Converting GT labels...", 0.1)

        # Convert GT labels to cleanlab format
        self.labels, self.image_paths = prepare_cleanlab_labels(
            self.images_path,
            self.gt_labels_path,
            filtered_paths
        )
        step_time = time.time() - step_start
        logger.info(f"Prepared {len(self.labels)} valid samples (耗时: {step_time:.3f}s)")

        step_start = time.time()
        self._report_progress("Counting classes...", 0.2)

        # Count classes
        self.num_classes = count_classes(
            self.gt_labels_path,
            self.pred_labels_path,
            self.image_paths
        )
        step_time = time.time() - step_start
        logger.info(f"Detected {self.num_classes} classes (耗时: {step_time:.3f}s)")

        step_start = time.time()
        self._report_progress("Converting Pred labels...", 0.3)

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
