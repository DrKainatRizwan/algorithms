"""Planar computational geometry: convex hulls, Delaunay triangulation, Voronoi diagrams."""

from .primitives import orient, in_circle, circumcenter, polygon_area, dist
from .hull import (
    monotone_chain,
    graham_scan,
    jarvis_march,
    quickhull,
    convex_hull,
    hull_area,
    hull_perimeter,
    diameter,
    point_in_convex,
)
from .delaunay import Delaunay, brute_force_delaunay
from .voronoi import Voronoi, clip_halfplane
from .applications import (
    euclidean_mst,
    closest_pair,
    largest_empty_circle,
    nearest_site,
)

__all__ = [
    "orient", "in_circle", "circumcenter", "polygon_area", "dist",
    "monotone_chain", "graham_scan", "jarvis_march", "quickhull", "convex_hull",
    "hull_area", "hull_perimeter", "diameter", "point_in_convex",
    "Delaunay", "brute_force_delaunay", "Voronoi", "clip_halfplane",
    "euclidean_mst", "closest_pair", "largest_empty_circle", "nearest_site",
]
