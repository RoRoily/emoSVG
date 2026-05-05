"""
Converts binary masks to SVG path data via contour extraction + cubic Bezier fitting.

Algorithm:
1. Extract contour points from the binary mask (OpenCV findContours).
2. Fit cubic Bezier curves to the contour using the Schneider algorithm
   (iterative least-squares tangent estimation).
3. Serialise the result as an SVG path data string ("M ... C ... Z").
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def _chord_length_parameterise(points: np.ndarray) -> np.ndarray:
    """Assign parameter t in [0,1] proportional to cumulative chord length."""
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumulative[-1]
    if total < 1e-9:
        return np.linspace(0.0, 1.0, len(points))
    return cumulative / total


def _cubic_bezier_point(p0, p1, p2, p3, t: float) -> np.ndarray:
    u = 1.0 - t
    return (u**3 * p0 + 3 * u**2 * t * p1
            + 3 * u * t**2 * p2 + t**3 * p3)


def _fit_cubic(points: np.ndarray, t_hat1: np.ndarray, t_hat2: np.ndarray):
    """
    Fit a single cubic Bezier to `points` with endpoint tangents t_hat1/t_hat2.
    Returns (P0, P1, P2, P3).
    """
    n = len(points)
    params = _chord_length_parameterise(points)

    # Build A matrix (least-squares)
    A = np.zeros((n, 2, 2))
    for i, t in enumerate(params):
        u = 1.0 - t
        b1 = 3 * u**2 * t
        b2 = 3 * u * t**2
        A[i][0] = b1 * t_hat1
        A[i][1] = b2 * t_hat2

    C = np.zeros((2, 2))
    X = np.zeros(2)
    p0, p3 = points[0], points[-1]

    for i, t in enumerate(params):
        u = 1.0 - t
        b0 = u**3
        b3 = t**3
        tmp = points[i] - (b0 * p0 + b3 * p3)
        C[0][0] += np.dot(A[i][0], A[i][0])
        C[0][1] += np.dot(A[i][0], A[i][1])
        C[1][0] = C[0][1]
        C[1][1] += np.dot(A[i][1], A[i][1])
        X[0] += np.dot(A[i][0], tmp)
        X[1] += np.dot(A[i][1], tmp)

    det_C = C[0][0] * C[1][1] - C[0][1] * C[1][0]
    det_X0 = X[0] * C[1][1] - C[0][1] * X[1]
    det_X1 = C[0][0] * X[1] - X[0] * C[1][0]

    alpha1 = det_X0 / det_C if abs(det_C) > 1e-12 else 0.0
    alpha2 = det_X1 / det_C if abs(det_C) > 1e-12 else 0.0

    seg_len = _distance(p0, p3)
    eps = 1e-6 * seg_len
    if alpha1 < eps or alpha2 < eps:
        alpha1 = alpha2 = seg_len / 3.0

    p1 = p0 + alpha1 * t_hat1
    p2 = p3 + alpha2 * t_hat2
    return p0, p1, p2, p3


def _max_error(points: np.ndarray, bezier, params: np.ndarray):
    """Return (max_error, split_index) for the fitted curve."""
    max_err = 0.0
    split = len(points) // 2
    for i, (pt, t) in enumerate(zip(points, params)):
        p0, p1, p2, p3 = bezier
        diff = pt - _cubic_bezier_point(p0, p1, p2, p3, t)
        err = float(np.dot(diff, diff))
        if err > max_err:
            max_err = err
            split = i
    return math.sqrt(max_err), split


def _fit_curve(
    points: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
    tolerance: float,
    depth: int = 0,
    max_depth: int = 8,
) -> list[tuple]:
    """Recursively fit cubic Bezier segments to a polyline."""
    if len(points) < 2:
        return []
    if len(points) == 2:
        dist = _distance(points[0], points[1]) / 3.0
        p1 = points[0] + dist * t_hat1
        p2 = points[1] + dist * t_hat2
        return [(points[0], p1, p2, points[1])]

    bezier = _fit_cubic(points, t_hat1, t_hat2)
    params = _chord_length_parameterise(points)
    max_err, split_idx = _max_error(points, bezier, params)

    if max_err < tolerance or depth >= max_depth:
        return [bezier]

    # Split and recurse
    t_hat_centre = points[split_idx - 1] - points[split_idx + 1]
    norm = np.linalg.norm(t_hat_centre)
    if norm > 1e-9:
        t_hat_centre /= norm

    left = _fit_curve(
        points[:split_idx + 1], t_hat1, t_hat_centre, tolerance, depth + 1, max_depth
    )
    right = _fit_curve(
        points[split_idx:], -t_hat_centre, t_hat2, tolerance, depth + 1, max_depth
    )
    return left + right


def _tangent_at_start(points: np.ndarray) -> np.ndarray:
    t = points[1] - points[0]
    n = np.linalg.norm(t)
    return t / n if n > 1e-9 else np.array([1.0, 0.0])


def _tangent_at_end(points: np.ndarray) -> np.ndarray:
    t = points[-2] - points[-1]
    n = np.linalg.norm(t)
    return t / n if n > 1e-9 else np.array([-1.0, 0.0])


def _segments_to_path_data(segments: list[tuple], precision: int) -> str:
    """Serialise a list of (P0,P1,P2,P3) tuples to SVG path data."""
    if not segments:
        return ""
    fmt = f".{precision}f"
    p0 = segments[0][0]
    parts = [f"M {p0[0]:{fmt}},{p0[1]:{fmt}}"]
    for _, p1, p2, p3 in segments:
        parts.append(
            f"C {p1[0]:{fmt}},{p1[1]:{fmt}} "
            f"{p2[0]:{fmt}},{p2[1]:{fmt}} "
            f"{p3[0]:{fmt}},{p3[1]:{fmt}}"
        )
    parts.append("Z")
    return " ".join(parts)


class BezierFitter:
    """
    Converts a binary mask to one or more SVG path data strings.

    Parameters
    ----------
    tolerance:      Maximum allowed pixel error per Bezier segment.
    max_nodes:      Discard contours with more points than this (noise guard).
    smooth_corners: Apply corner smoothing before fitting.
    precision:      Decimal places in SVG coordinate output.
    """

    def __init__(
        self,
        tolerance: float = 1.5,
        max_nodes: int = 200,
        smooth_corners: bool = True,
        precision: int = 2,
    ) -> None:
        self.tolerance = tolerance
        self.max_nodes = max_nodes
        self.smooth_corners = smooth_corners
        self.precision = precision

    def mask_to_paths(self, mask: np.ndarray) -> list[str]:
        """
        Convert a boolean HxW mask to a list of SVG path data strings
        (one string per connected contour).
        """
        binary = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        paths: list[str] = []
        for contour in contours:
            pts = contour.squeeze()
            if pts.ndim != 2 or len(pts) < 4:
                continue
            if len(pts) > self.max_nodes:
                # Downsample to max_nodes evenly
                idx = np.round(np.linspace(0, len(pts) - 1, self.max_nodes)).astype(int)
                pts = pts[idx]

            points = pts.astype(np.float64)
            if self.smooth_corners:
                points = self._smooth(points)

            t1 = _tangent_at_start(points)
            t2 = _tangent_at_end(points)
            segments = _fit_curve(points, t1, t2, self.tolerance)
            if segments:
                paths.append(_segments_to_path_data(segments, self.precision))

        return paths

    @staticmethod
    def _smooth(points: np.ndarray, window: int = 3) -> np.ndarray:
        """Simple moving-average smoothing of contour points."""
        n = len(points)
        if n <= window:
            return points
        kernel = np.ones(window) / window
        smoothed = np.stack([
            np.convolve(points[:, 0], kernel, mode="same"),
            np.convolve(points[:, 1], kernel, mode="same"),
        ], axis=1)
        # Preserve exact endpoints
        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
        return smoothed
