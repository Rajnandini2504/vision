
import streamlit as st
import cv2
import numpy as np


st.set_page_config(
    page_title="Vision Shield AI",
    layout="wide"
)

st.title("Vision Shield AI")
st.write("Road Lane Detection and Vehicle Headlight Glare Analysis")


# 1. LANE DETECTION

def detect_lanes(image):
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height - 1),
        (width - 1, height - 1),
        (int(width * 0.9), int(height * 0.55)),
        (int(width * 0.1), int(height * 0.55))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        60,
        minLineLength=max(25, int(height * 0.08)),
        maxLineGap=max(15, int(height * 0.04))
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / float(dx)
            angle = abs(np.degrees(np.arctan(slope)))
            mid_x = (x1 + x2) / 2

            if not 25 <= angle <= 75:
                continue

            if max(y1, y2) < height * 0.60:
                continue

            if slope < 0 and mid_x < width * 0.60:
                left_lines.append((x1, y1, x2, y2))

            elif slope > 0 and mid_x > width * 0.40:
                right_lines.append((x1, y1, x2, y2))

    for line in left_lines + right_lines:
        x1, y1, x2, y2 = line
        cv2.line(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3
        )

    
    # Check whether left and right line candidates meet
    # near a plausible vanishing area.
    left_detected = len(left_lines) > 0
    right_detected = len(right_lines) > 0
    left_horizon = []
    right_horizon = []
