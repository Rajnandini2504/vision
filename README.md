# Vision Shield AI – Lane + Glare Safety

## Project Overview
Vision Shield AI is a Python-based image-processing application designed to analyze road lane markings and bright regions that may correspond to vehicle headlights. The application uses computer vision techniques to provide preliminary visual analysis of uploaded images.

## Objectives
- Identify potential road lane markings.
- Detect paired bright regions that may represent vehicle headlights.
- Analyze image brightness in detected headlight regions.
- Reject images when neither lane markings nor valid headlight candidates are detected.

## Key Features
- Image upload through a Streamlit web interface.
- Lane marking analysis using OpenCV image-processing techniques.
- Heuristic detection of potential paired headlights.
- Estimated brightness index for detected headlight regions.
- Processed image visualization and detection results.
- Basic validation to reject unrelated images.

## Technologies Used
- Python
- Streamlit
- OpenCV
- NumPy
- Pillow

## How It Works
1. The user uploads an image.
2. The application processes the image using computer vision techniques.
3. Lane marking candidates and potential headlight regions are analyzed separately.
4. The application displays the detection results and an estimated brightness index when applicable.
5. Images that do not satisfy either detection check are rejected.

## Installation
1. Clone this repository:
   `git clone https://github.com/Rajnandini2504/vision.git`
2. Open the project folder:
   `cd vision`
3. Install the dependencies:
   `pip install -r requirements.txt`
4. Run the application:
   `streamlit run app.py`

## Limitations
This project uses rule-based image processing. Bright objects may be mistaken for headlights, and road-like edges may be mistaken for lane markings. The brightness index is an estimate based on image pixels; it is not a calibrated measurement of real-world glare. Results depend on image quality, lighting, perspective, and detection thresholds.

This application is intended for educational and experimental use. It is not a certified vehicle safety system.

## Author
Rajnandini Thalkari

## License
Copyright and licensing status should be specified according to the project's actual ownership and applicable registration or license terms.
