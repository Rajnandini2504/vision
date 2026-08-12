import os
import uuid

import streamlit as st

try:
    import cv2
except ModuleNotFoundError:
    st.error(
        "OpenCV is not installed in this environment (missing dependency: 'cv2').\n\n"
        "Install it with:\n"
        "  pip install opencv-python\n\n"
        "Then restart the app."
    )
    st.stop()

import numpy as np


def process_image_bgr(image_bgr: np.ndarray):
    output = image_bgr.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


    # GLARE DETECTION
    lower = np.array([0, 0, 200])
    upper = np.array([180, 60, 255])

    glare_mask = cv2.inRange(hsv, lower, upper)
    glare_pixels = np.sum(glare_mask > 0)
    total_pixels = glare_mask.size
    glare_percentage = (glare_pixels / total_pixels) * 100

    # IMAGE ENHANCEMENT
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # EDGE DETECTION
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # REGION OF INTEREST (Only lower half for lanes)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array(
        [
            [
                (0, height),
                (width, height),
                (width, int(height * 0.55)),
                (0, int(height * 0.55)),
            ]
        ],
        np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # LANE DETECTION
    lines = cv2.HoughLinesP(
        cropped_edges,
        1,
        np.pi / 180,
        80,
        minLineLength=60,
        maxLineGap=20,
    )

    lane_confidence = 0
    valid_lane_count = 0

    if lines is not None:
        # OpenCV may return lines as either (N, 1, 4) or (N, 4), depending on
        # the build. Normalize both forms before unpacking coordinates.
        line_segments = np.asarray(lines).reshape(-1, 4)
        for x1, y1, x2, y2 in line_segments:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            slope = 0
            if (x2 - x1) != 0:
                slope = (y2 - y1) / (x2 - x1)

            # Accept only valid lane slopes
            if abs(slope) > 0.5:
                valid_lane_count += 1
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

        lane_confidence = min(valid_lane_count * 10, 100)

    # STATUS LOGIC
    # Requirement: after upload + image, if vehicle/lane is not detected then show "Image not found".
    # We approximate vehicle/lane absence using lane confidence threshold.
    lane_present = lane_confidence >= 30

    status = "SAFE"
    status_color = (0, 255, 0)

    if glare_percentage > 18:
        status = "UNSAFE - HIGH GLARE"
        status_color = (0, 0, 255)
    elif not lane_present:
        status = "Image not found"
        status_color = (0, 0, 255)
    elif glare_percentage > 10 or lane_confidence < 50:
        status = "WARNING"
        status_color = (0, 255, 255)


    # TEXT OVERLAY
    cv2.putText(
        output,
        f"Glare: {glare_percentage:.2f}%",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        output,
        f"Lane Confidence: {lane_confidence}%",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        output,
        f"Status: {status}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3,
    )

    return output, glare_percentage, lane_confidence, status


st.set_page_config(page_title="Vision Shield AI", layout="centered")
st.title("Vision Shield AI - Lane + Glare Safety")

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload image to get processed result.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    st.error("Could not read the uploaded image.")
    st.stop()

output_bgr, glare_percentage, lane_confidence, status = process_image_bgr(image_bgr)

# Save to a temp output folder (so Streamlit can show it)
os.makedirs("outputs/processed_images", exist_ok=True)
output_path = os.path.join("outputs/processed_images", f"{uuid.uuid4().hex}.png")
cv2.imwrite(output_path, output_bgr)

# Display
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    st.image(image_bgr[:, :, ::-1], channels="RGB")

with col2:
    st.subheader("Output")
    st.image(output_bgr[:, :, ::-1], channels="RGB")

st.success(status)
st.write({"Glare (%)": float(glare_percentage), "Lane Confidence (%)": int(lane_confidence)})

