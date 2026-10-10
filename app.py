
import streamlit as st
import cv2
import numpy as np


st.set_page_config(
    page_title="Vision Shield AI",
    page_icon="🚘",
    layout="wide",
)

# --------------------------------------------------
# 1. LANE DETECTION
# --------------------------------------------------

def detect_lanes(image_bgr):
    output = image_bgr.copy()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height, width = edges.shape

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height - 1),
        (width - 1, height - 1),
        (int(width * 0.90), int(height * 0.55)),
        (int(width * 0.10), int(height * 0.55)),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        cropped_edges,
        1,
        np.pi / 180,
        60,
        minLineLength=max(25, int(height * 0.08)),
        maxLineGap=max(15, int(height * 0.04)),
    )

    valid_segments = []

    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx = x2 - x1

            if dx == 0:
                continue

            slope = (y2 - y1) / float(dx)
            angle = abs(np.degrees(np.arctan(slope)))

            # Reject horizontal and nearly vertical edges.
            if 25 <= angle <= 75:
                if max(y1, y2) >= height * 0.60:
                    valid_segments.append(
                        (int(x1), int(y1), int(x2), int(y2))
                    )

    # Draw candidate lane segments.
    for x1, y1, x2, y2 in valid_segments:
        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

    # Heuristic score, NOT a trained-model probability.
    lane_confidence = min(len(valid_segments) * 10, 100)

    # Require repeated line evidence to reduce false positives.
    lane_present = len(valid_segments) >= 2

    return output, lane_confidence, lane_present


# --------------------------------------------------
# 2. VEHICLE HEADLIGHT DETECTION
# --------------------------------------------------

def detect_headlights(image_bgr):
    height, width = image_bgr.shape[:2]

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Bright, relatively neutral-coloured candidates.
    lower = np.array([0, 0, 215], dtype=np.uint8)
    upper = np.array([180, 110, 255], dtype=np.uint8)

    bright_mask = cv2.inRange(hsv, lower, upper)

    # Remove isolated noise.
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    contours, _ = cv2.findContours(
        bright_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        box_area = w * h
        ratio = w / float(max(h, 1))

        # Reject tiny noise and excessively large bright regions.
        if (
            area >= 5
            and box_area <= height * width * 0.05
            and 0.30 <= ratio <= 3.0
        ):
            candidates.append((x, y, w, h, contour))

    # Search for two horizontally separated, similar-sized lights.
    selected = []

    for i, first in enumerate(candidates):
        x1, y1, w1, h1, _ = first
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2

        for second in candidates[i + 1:]:
            x2, y2, w2, h2, _ = second
            center2_x = x2 + w2 / 2
            center2_y = y2 + h2 / 2

            gap = abs(center1_x - center2_x)

            same_level = (
                abs(center1_y - center2_y)
                <= max(h1, h2) * 1.5
            )

            similar_size = (
                0.35
                <= (w1 * h1) / float(max(w2 * h2, 1))
                <= 2.85
            )

            reasonable_gap = (
                max(w1, w2) * 1.2
                <= gap
                <= width * 0.60
            )

            if same_level and similar_size and reasonable_gap:
                selected.extend([first, second])

    # Deduplicate selected bounding boxes.
    unique_boxes = {}
    for x, y, w, h, contour in selected:
        unique_boxes[(x, y, w, h)] = contour

    headlight_boxes = list(unique_boxes.keys())

    headlight_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for x, y, w, h in headlight_boxes:
        cv2.rectangle(
            headlight_mask,
            (x, y),
            (x + w, y + h),
            255,
            thickness=cv2.FILLED,
        )

    headlights_present = len(headlight_boxes) >= 2

    return headlight_boxes, headlight_mask, headlights_present


# --------------------------------------------------
# 3. IMAGE PROCESSING
# --------------------------------------------------

def process_image(image_bgr):
    output, lane_confidence, lane_present = detect_lanes(
        image_bgr
    )

    headlight_boxes, headlight_mask, headlights_present = (
        detect_headlights(image_bgr)
    )

    # Count bright pixels across the image.
    # This is a brightness indicator, not a certified
    # measurement of optical glare.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    bright_pixels = cv2.inRange(
        hsv,
        np.array([0, 0, 200], dtype=np.uint8),
        np.array([180, 110, 255], dtype=np.uint8),
    )

    bright_pixel_percentage = (
        np.count_nonzero(bright_pixels)
        / bright_pixels.size
    ) * 100

    # Draw detected headlight bounding boxes.
    for x, y, w, h in headlight_boxes:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2,
        )

    # Valid if EITHER lane markings OR paired headlights
    # are detected. Both are not required.
    valid_image = lane_present or headlights_present

    if not valid_image:
        status = "INVALID IMAGE - NO LANE OR HEADLIGHTS"
        status_color = (0, 0, 255)

    elif bright_pixel_percentage > 18:
        status = "HIGH BRIGHT-PIXEL LEVEL - REVIEW"
        status_color = (0, 0, 255)

    elif lane_present and headlights_present:
        status = "LANE + HEADLIGHTS DETECTED"
        status_color = (0, 255, 0)

    elif lane_present:
        status = "LANE DETECTED"
        status_color = (0, 255, 0)

    else:
        status = "VEHICLE HEADLIGHTS DETECTED"
        status_color = (0, 255, 255)

    cv2.putText(
        output,
        f"Bright pixels: {bright_pixel_percentage:.2f}%",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        output,
        f"Lane score: {lane_confidence}%",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        output,
        f"Status: {status}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
    )

    return (
        output,
        bright_pixel_percentage,
        lane_confidence,
        lane_present,
        headlights_present,
        valid_image,
        status,
    )


# --------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("Vision Shield AI")
st.subheader("Road Lane and Vehicle Headlight Analysis")

st.write(
    "Upload a road image with visible lane markings, "
    "vehicle headlights, or both."
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("Please upload an image to begin.")
    st.stop()

file_bytes = np.frombuffer(
    uploaded.getvalue(),
    dtype=np.uint8,
)

image_bgr = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_COLOR,
)

if image_bgr is None:
    st.error("The uploaded file could not be decoded as an image.")
    st.stop()

(
    output_bgr,
    bright_pixel_percentage,
    lane_confidence,
    lane_present,
    headlights_present,
    valid_image,
    status,
) = process_image(image_bgr)

# Reject unrelated images instead of showing a normal
# detection result.
if not valid_image:
    st.error(
        "Invalid Image: No road lane markings or paired "
        "vehicle headlights were detected."
    )
    st.subheader("Uploaded Image")
    st.image(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )
    st.info(
        "Try a clearer image with visible road lane markings "
        "or a pair of vehicle headlights."
    )
    st.stop()

# Display results only for images passing the detection rules.
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

with col2:
    st.subheader("Processed Image")
    st.image(
        cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

st.subheader("Analysis Results")

st.success(status)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Lane Detection",
    "Detected" if lane_present else "Not detected",
)

m2.metric(
    "Headlights",
    "Detected" if headlights_present else "Not detected",
)

m3.metric(
    "Lane Heuristic Score",
    f"{lane_confidence}%",
)

st.metric(
    "Bright Pixel Ratio",
    f"{bright_pixel_percentage:.2f}%",
)

st.caption(
    "Bright Pixel Ratio is not a calibrated optical glare "
    "measurement. Lane score is a heuristic, not a probability. "
    "This prototype is not a certified vehicle safety system."
)
