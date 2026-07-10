"""
Procedural smiley face renderer used to visualize an ABSA sentiment score.

Given a positivity score in [-1, 1], draws a face whose color shifts from red
(negative) through yellow (neutral) to green (positive), and whose mouth curves
into a smile or a frown accordingly.

The overall idea (HSV color interpolation for the face + a curved mouth line)
is inspired by the open-source project
https://github.com/trflorian/sentiment-analysis-viz, reimplemented here for
this project's own ABSA pipeline.
"""

import cv2
import numpy as np


def _positivity_to_bgr(positivity: float) -> tuple[int, int, int]:
    """Maps positivity in [-1, 1] to a face color: red -> yellow -> green."""
    hue = (positivity + 1) / 2 * (1 / 3)  # 0 (red) .. 1/3 (green), through 1/6 (yellow)
    hsv_pixel = np.array([[[hue * 180, 140, 255]]], dtype=np.uint8)
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def render_smiley(positivity: float, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Renders a smiley face as an RGBA numpy array (BGR + alpha channel order).

    Args:
        positivity: Sentiment score in [-1, 1], -1 being very negative and 1 very positive.
        size: (width, height) of the output image in pixels.

    Returns:
        A (height, width, 4) uint8 numpy array in BGRA order.
    """
    positivity = float(np.clip(positivity, -1.0, 1.0))
    width, height = size
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    outline_color = (60, 60, 60, 255)
    outline_thickness = max(2, min(size) // 30)
    center = (width // 2, height // 2)
    radius = min(size) // 2 - outline_thickness

    face_bgr = _positivity_to_bgr(positivity)
    face_bgra = (*face_bgr, 255)

    cv2.circle(frame, center, radius, face_bgra, thickness=-1)
    cv2.circle(frame, center, radius, outline_color, outline_thickness)

    # eyes
    eye_radius = radius // 5
    eye_dx = radius // 3
    eye_dy = radius // 4
    for sign in (-1, 1):
        eye_center = (center[0] + sign * eye_dx, center[1] - eye_dy)
        cv2.circle(frame, eye_center, eye_radius, outline_color, thickness=-1)

    # mouth: a curve pinned at the corners, bending up (smile) or down (frown)
    # in proportion to the positivity score.
    mouth_half_width = radius // 2
    mouth_bend = radius // 3
    mouth_baseline_y = center[1] + radius // 3

    t = np.linspace(-1, 1, 100)
    bend = positivity * mouth_bend * (1 - t**2)

    points = np.stack(
        [
            center[0] + t * mouth_half_width,
            mouth_baseline_y + bend,
        ],
        axis=1,
    ).astype(np.int32)

    cv2.polylines(
        frame,
        [points],
        isClosed=False,
        color=outline_color,
        thickness=int(outline_thickness * 1.5),
    )

    return frame
