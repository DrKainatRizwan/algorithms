"""Interval trees and segment trees for computational-geometry queries."""
from .bench import random_intervals, random_rects, timeit
from .interval_tree import CenteredIntervalTree, DynamicIntervalTree, SortedEndpointIndex
from .segment_tree import CoverageTree, MaxAddTree
from .sweep import (RectangleIndex, max_overlap_point, rect_intersections, skyline,
                    union_area, union_length)

__all__ = [
    "CenteredIntervalTree", "DynamicIntervalTree", "SortedEndpointIndex",
    "CoverageTree", "MaxAddTree", "RectangleIndex", "max_overlap_point",
    "rect_intersections", "skyline", "union_area", "union_length",
    "random_intervals", "random_rects", "timeit",
]
