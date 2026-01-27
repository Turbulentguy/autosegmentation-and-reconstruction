"""
Metrics Module for MySpineSAM3
==============================
Implements DSC, IoU, HD95, ASD, and InferenceTime metrics.
Uses medpy for surface distance calculations.
"""

import time
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-7) -> float:
    """Compute Dice Similarity Coefficient."""
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-7) -> float:
    """Compute Intersection over Union."""
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + smooth) / (union + smooth)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray, voxel_spacing: tuple = (1,1,1)) -> float:
    """Compute 95th percentile Hausdorff Distance using medpy."""
    try:
        from medpy.metric.binary import hd95
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return float('inf')
        return hd95(pred.astype(bool), target.astype(bool), voxelspacing=voxel_spacing)
    except ImportError:
        logger.warning("medpy not installed, HD95 unavailable")
        return 0.0


def average_surface_distance(pred: np.ndarray, target: np.ndarray, voxel_spacing: tuple = (1,1,1)) -> float:
    """Compute Average Surface Distance using medpy."""
    try:
        from medpy.metric.binary import asd
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return float('inf')
        return asd(pred.astype(bool), target.astype(bool), voxelspacing=voxel_spacing)
    except ImportError:
        logger.warning("medpy not installed, ASD unavailable")
        return 0.0


class InferenceTimer:
    """Context manager for timing inference."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


class MetricCalculator:
    """Calculate and aggregate segmentation metrics."""
    
    def __init__(self, metrics: List[str] = None, include_background: bool = False, num_classes: int = 2):
        self.metrics = metrics or ["DSC", "IOU", "HD95", "ASD"]
        self.include_background = include_background
        self.num_classes = num_classes
        self.results = []
    
    def compute(self, pred: np.ndarray, target: np.ndarray, 
                voxel_spacing: tuple = (1,1,1), inference_time_ms: float = 0) -> Dict[str, float]:
        """Compute all metrics for a single prediction.
        
        Only computes metrics for classes that actually exist in the ground truth,
        which avoids inflated scores from empty classes.
        """
        result = {}
        start_class = 0 if self.include_background else 1
        
        # Ensure integer types for class comparison (fixes float mask interpolation)
        pred = pred.astype(np.int32)
        target = target.astype(np.int32)
        
        # Find classes that actually exist in ground truth
        existing_classes = [c for c in range(start_class, self.num_classes) if np.any(target == c)]
        
        # Debug: log class info on first call
        if len(self.results) == 0:
            unique_target = np.unique(target)
            unique_pred = np.unique(pred)
            logger.info(f"[MetricCalculator] Target unique classes: {unique_target.tolist()}")
            logger.info(f"[MetricCalculator] Pred unique classes: {unique_pred.tolist()}")
            logger.info(f"[MetricCalculator] existing_classes (excluding bg): {existing_classes}")
        
        if not existing_classes:
            # No valid classes found
            for metric in self.metrics:
                if metric == "InferenceTime":
                    result["InferenceTime"] = inference_time_ms
                else:
                    result[metric] = 0.0
            self.results.append(result)
            return result
        
        for metric in self.metrics:
            if metric == "DSC":
                scores = []
                for c in existing_classes:
                    pred_c = (pred == c).astype(float)
                    target_c = (target == c).astype(float)
                    score = dice_coefficient(pred_c, target_c)
                    scores.append(score)
                result["DSC"] = np.mean(scores) if scores else 0.0
            elif metric == "IOU":
                scores = []
                for c in existing_classes:
                    pred_c = (pred == c).astype(float)
                    target_c = (target == c).astype(float)
                    score = iou_score(pred_c, target_c)
                    scores.append(score)
                result["IOU"] = np.mean(scores) if scores else 0.0
            elif metric == "HD95":
                scores = []
                for c in existing_classes:
                    pred_c = (pred == c).astype(int)
                    target_c = (target == c).astype(int)
                    # Only compute if both pred and target have this class
                    if np.any(pred_c) and np.any(target_c):
                        scores.append(hausdorff_distance_95(pred_c, target_c, voxel_spacing))
                valid_scores = [s for s in scores if s != float('inf') and not np.isnan(s)]
                result["HD95"] = np.mean(valid_scores) if valid_scores else float('nan')
            elif metric == "ASD":
                scores = []
                for c in existing_classes:
                    pred_c = (pred == c).astype(int)
                    target_c = (target == c).astype(int)
                    # Only compute if both pred and target have this class
                    if np.any(pred_c) and np.any(target_c):
                        scores.append(average_surface_distance(pred_c, target_c, voxel_spacing))
                valid_scores = [s for s in scores if s != float('inf') and not np.isnan(s)]
                result["ASD"] = np.mean(valid_scores) if valid_scores else float('nan')
            elif metric == "InferenceTime":
                result["InferenceTime"] = inference_time_ms
        
        self.results.append(result)
        return result
    
    def get_average(self) -> Dict[str, float]:
        if not self.results: return {}
        return {k: np.mean([r[k] for r in self.results]) for k in self.results[0]}
    
    def get_std(self) -> Dict[str, float]:
        if not self.results: return {}
        return {k: np.std([r[k] for r in self.results]) for k in self.results[0]}
