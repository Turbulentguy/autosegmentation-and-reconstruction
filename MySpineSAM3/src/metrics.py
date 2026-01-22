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
        """Compute all metrics for a single prediction."""
        result = {}
        start_class = 0 if self.include_background else 1
        
        for metric in self.metrics:
            if metric == "DSC":
                scores = [dice_coefficient((pred==c).astype(float), (target==c).astype(float)) 
                         for c in range(start_class, self.num_classes)]
                result["DSC"] = np.mean(scores)
            elif metric == "IOU":
                scores = [iou_score((pred==c).astype(float), (target==c).astype(float))
                         for c in range(start_class, self.num_classes)]
                result["IOU"] = np.mean(scores)
            elif metric == "HD95":
                scores = [hausdorff_distance_95((pred==c).astype(int), (target==c).astype(int), voxel_spacing)
                         for c in range(start_class, self.num_classes)]
                result["HD95"] = np.mean([s for s in scores if s != float('inf')] or [0])
            elif metric == "ASD":
                scores = [average_surface_distance((pred==c).astype(int), (target==c).astype(int), voxel_spacing)
                         for c in range(start_class, self.num_classes)]
                result["ASD"] = np.mean([s for s in scores if s != float('inf')] or [0])
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
