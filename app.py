
import streamlit as st
import cv2
import numpy as np

# Optional pretrained object detector. Headlights are accepted only when a
# vehicle is also detected; this helps reject lamps and bright windows.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


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

            # Generic road-lane checks: require long, lower-frame
            # segments with a plausible lane-marking angle.
            line_length = float(np.hypot(dx, dy))
            if not 30 <= angle <= 70:
                continue

            if line_length < height * 0.15:
                continue

            if max(y1, y2) < height * 0.65:
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

    for x1, y1, x2, y2 in left_lines:
        slope = (y2 - y1) / float(x2 - x1)
        if abs(slope) > 0.001:
            horizon_x = x1 + (height * 0.55 - y1) / slope
            left_horizon.append(horizon_x)

    for x1, y1, x2, y2 in right_lines:
        slope = (y2 - y1) / float(x2 - x1)
        if abs(slope) > 0.001:
            horizon_x = x1 + (height * 0.55 - y1) / slope
            right_horizon.append(horizon_x)

    lane_present = False

    if (
        len(left_lines) >= 2
        and len(right_lines) >= 2
        and left_horizon
        and right_horizon
    ):
        left_x = float(np.median(left_horizon))
        right_x = float(np.median(right_horizon))

        lane_present = (
            0.20 * width <= left_x <= 0.80 * width
            and 0.20 * width <= right_x <= 0.80 * width
            and abs(left_x - right_x) <= 0.20 * width
            and left_x < right_x
        )

    # Evidence score only; it is not a road-safety percentage.
    if lane_present:
        lane_score = 100
        lane_status = "BOTH LANE SIDES DETECTED - GEOMETRY CHECK PASSED"
    elif left_detected and right_detected:
        lane_score = 25
        lane_status = "LINES FOUND BUT LANE GEOMETRY NOT VERIFIED"
    elif left_detected or right_detected:
        lane_score = 50
        lane_status = "PARTIAL LANE CANDIDATE - WARNING"
    else:
        lane_score = 0
        lane_status = "LANE NOT DETECTED"


    return output, lane_present, lane_score, lane_status


# 2. VEHICLE + HEADLIGHT DETECTION

@st.cache_resource
def load_vehicle_model():
    """Load a small pretrained COCO detector once per Streamlit session."""
    if YOLO is None:
        return None
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        return None


def detect_vehicles(image):
    """Return vehicle boxes; fail closed if the detector is unavailable."""
    model = load_vehicle_model()
    if model is None:
        return []

    try:
        # COCO classes: car=2, motorcycle=3, bus=5, truck=7.
        results = model.predict(
            source=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            classes=[2, 3, 5, 7],
            conf=0.35,
            verbose=False
        )
        boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [int(v) for v in box]
                boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception:
        return []


def detect_headlights(image, vehicle_boxes):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 215], dtype=np.uint8)
    upper = np.array([180, 110, 255], dtype=np.uint8)
    bright_mask = cv2.inRange(hsv, lower, upper)
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # No detected vehicle = no headlight detection. This is the key fix for
    # indoor lamps/windows being mistaken for headlights.
    if not vehicle_boxes:
        return [], np.zeros((height, width), dtype=np.uint8), False

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < max(20, int(height * width * 0.00008)):
            continue
        if w * h > height * width * 0.02:
            continue
        if w < width * 0.008 or h < height * 0.008:
            continue
        if y < height * 0.30 or (y + h) > height * 0.95:
            continue
        ratio = w / float(max(h, 1))
        if not 0.35 <= ratio <= 2.8:
            continue

        cx, cy = x + w / 2.0, y + h / 2.0
        # Candidate must lie inside a detected vehicle box (with a small margin).
        belongs_to_vehicle = any(
            x1 - 0.03 * width <= cx <= x2 + 0.03 * width
            and y1 - 0.03 * height <= cy <= y2 + 0.03 * height
            for x1, y1, x2, y2 in vehicle_boxes
        )
        if belongs_to_vehicle:
            candidates.append((x, y, w, h))

    pairs = []
    for i, first in enumerate(candidates):
        x1, y1, w1, h1 = first
        c1x, c1y = x1 + w1 / 2.0, y1 + h1 / 2.0
        for second in candidates[i + 1:]:
            x2, y2, w2, h2 = second
            c2x, c2y = x2 + w2 / 2.0, y2 + h2 / 2.0
            gap = abs(c1x - c2x)
            same_level = abs(c1y - c2y) <= max(h1, h2) * 1.5
            similar_size = 0.35 <= (w1 * h1) / float(max(w2 * h2, 1)) <= 2.85
            reasonable_gap = max(w1, w2) * 2.0 <= gap <= width * 0.35
            pair_center = (c1x + c2x) / 2.0
            centered_pair = width * 0.10 <= pair_center <= width * 0.90
            if same_level and similar_size and reasonable_gap and centered_pair:
                # Both candidates must belong to the same detected vehicle.
                same_vehicle = any(
                    x1b - 0.03 * width <= c1x <= x2b + 0.03 * width
                    and x1b - 0.03 * width <= c2x <= x2b + 0.03 * width
                    and y1b - 0.03 * height <= c1y <= y2b + 0.03 * height
                    and y1b - 0.03 * height <= c2y <= y2b + 0.03 * height
                    for x1b, y1b, x2b, y2b in vehicle_boxes
                )
                if same_vehicle:
                    pairs.append((first, second))

    selected = list(pairs[0]) if pairs else []
    headlight_mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in selected:
        cv2.rectangle(headlight_mask, (x, y), (x + w, y + h), 255, -1)
    return selected, headlight_mask, len(selected) == 2


# 3. PROCESS THE IMAGE

def process_image(image):
    output, lane_present, lane_score, lane_status = (
        detect_lanes(image)
    )

    # Use a copy because headlight drawing must not change input.
    headlight_input = image.copy()

    vehicle_boxes = detect_vehicles(headlight_input)
    headlights, headlight_mask, headlights_present = (
        detect_headlights(headlight_input, vehicle_boxes)
    )

    # Measure brightness inside the detected headlight regions.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2]

    if headlights_present:
        values = []

        for x, y, w, h in headlights:
            values.extend(
                brightness[y:y + h, x:x + w].flatten().tolist()
            )

        
        if values:
            pixel_values = np.asarray(values, dtype=np.float32)
        
            # Average brightness of detected headlight regions
            mean_brightness = np.mean(pixel_values) / 255 * 100
        
            # Percentage of pixels that are nearly white
            very_bright_percentage = (
                np.count_nonzero(pixel_values >= 240)
                / pixel_values.size
            ) * 100
        
            # Experimental image-based brightness indicator
            glare_score = (
                0.70 * mean_brightness
                + 0.30 * very_bright_percentage
            )
        else:
            glare_score = 0


        for x, y, w, h in headlights:
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )
    else:
        glare_score = 0

    valid_image = lane_present or headlights_present

    if not valid_image:
        overall_status = "INVALID IMAGE"
        glare_status = "HEADLIGHTS NOT DETECTED"
        lane_status = "LANE NOT CLEARLY DETECTED"

    else:
        if headlights_present:
            if glare_score >= 85:
                glare_status = "HIGH BRIGHTNESS - UNSAFE WARNING"
            elif glare_score >= 65:
                glare_status = "MODERATE BRIGHTNESS - WARNING"
            else:
                glare_status = "LOWER BRIGHTNESS - REVIEW"

        else:
            glare_status = "HEADLIGHTS NOT DETECTED"

        if lane_present:
            lane_status = "LANE DETECTED - REVIEW ALIGNMENT"
        else:
            lane_status = "LANE NOT CLEARLY DETECTED"

        if headlights_present and glare_score >= 85:
            overall_status = "WARNING - HIGH HEADLIGHT BRIGHTNESS"
        elif lane_present and headlights_present:
            overall_status = "LANE + HEADLIGHTS DETECTED"
        elif lane_present:
            overall_status = "LANE DETECTED"
        else:
            overall_status = "HEADLIGHTS DETECTED"

    cv2.putText(
        output,
        f"Lane score: {lane_score}%",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Headlight brightness: {glare_score:.1f}%",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )

    return (
        output,
        valid_image,
        lane_present,
        lane_score,
        lane_status,
        headlights_present,
        glare_score,
        glare_status,
        overall_status
    )


# 4. STREAMLIT USER INTERFACE

uploaded = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    data = np.frombuffer(
        uploaded.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Image could not be opened.")
        st.stop()

    (
        output,
        valid_image,
        lane_present,
        lane_score,
        lane_status,
        headlights_present,
        glare_score,
        glare_status,
        overall_status
    ) = process_image(image)

    if not valid_image:
        st.error(
            "Invalid image: no clear lane geometry or paired headlight-like "
            "regions detected. This image should not be analyzed as a road scene."
        )
        st.caption(
            f"Validation details — lane geometry: {lane_present}; "
            f"paired headlight candidates: {headlights_present}. "
            "These are rule-based checks, not a trained road-scene classifier."
        )
         st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )
        )
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            width="stretch"
        )

    with col2:
        st.subheader("Processed Image")
        st.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            width="stretch"
        )

    st.subheader("Detection Results")
    st.write("Overall status:", overall_status)

    m1, m2 = st.columns(2)

    m1.metric(
        "Lane Detection Score",
        f"{lane_score}%"
    )

    m2.metric(
        "Estimated Headlight Brightness Index",
         f"{glare_score:.1f}%"
    )

    st.write("Lane result:", lane_status)
    st.write("Headlight result:", glare_status)

    if YOLO is None:
        st.warning(
            "Vehicle detector package is unavailable. Headlight detection is disabled; "
            "install ultralytics from requirements.txt and redeploy."
        )

    st.caption(
        "Prototype only: lane detection uses image-processing heuristics and "
        "headlight candidates require a detected vehicle. Brightness is an estimate, "
        "not a calibrated glare measurement or certified road-safety decision."
    )
