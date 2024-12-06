# YOLO-based Image Manipulation Tool

## Overview
This project provides an interactive web tool for object detection and segmentation using YOLOv5 and YOLOv8 models. It allows users to apply various image manipulation techniques like Grayscale, Edge Detection, and Blur to the processed images. Built with **Streamlit**, this tool enables easy interaction and testing of YOLO models in real-time through a simple user interface.

## Features
- **YOLO Object Detection**: Choose between YOLOv5 or YOLOv8 for detecting and segmenting objects in the uploaded image.
- **Image Manipulation**: Apply image manipulation techniques like:
  - Grayscale Conversion
  - Edge Detection (Canny)
  - Blur (Gaussian)
- **Interactive UI**: Built with **Streamlit**, providing a smooth experience for uploading images, selecting models, and applying manipulations.

## Requirements
The application requires the following Python libraries:
- `cv2` (OpenCV)
- `numpy`
- `Pillow` (PIL)
- `ultralytics` (YOLO models)
- `streamlit`

You can install the dependencies using the `requirements.txt` file.

## Installation Instructions

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/yolo-image-manipulation.git
    cd yolo-image-manipulation
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have the YOLO model files (`yolov5xu.pt` and `yolov8x-seg.pt`) in the same directory. You can download them from the official YOLO repository or use your pre-trained models.

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

5. Open your browser and go to `http://localhost:8501` to start using the tool.

## Usage

1. **Upload an Image**: Select an image in `.jpg`, `.jpeg`, or `.png` format.
2. **Choose a YOLO Model**: Pick between YOLOv5 or YOLOv8 for object detection.
3. **Choose Manipulation Technique**: Select an image manipulation technique (None, Grayscale, Edge Detection, or Blur).
4. **View the Results**: The tool will process the image and display the manipulated result in real-time.

## Example Workflow
1. Upload an image (e.g., an image of a street with cars and pedestrians).
2. Choose **YOLOv5** for detection.
3. Select **Grayscale** as the manipulation technique.
4. See the processed image with object detection bounding boxes and the grayscale effect applied.


