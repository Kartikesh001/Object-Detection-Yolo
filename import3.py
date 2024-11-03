import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# Load the YOLOv8 segmentation model
model_v8 = YOLO('yolov8x-seg.pt') 

# Function to process the image using YOLOv8 for detection and segmentation
def process_image(image):
    
    resized_image = image.resize((640, 640))
    resized_image_np = np.array(resized_image)

   
    results_v8 = model_v8.predict(resized_image_np)

    # Extract segmentation and detection results
    for r in results_v8[0].boxes.data.tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = r
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        label = model_v8.names[int(class_id)]  # Get label name from class_id

        # Drawing bounding box using OpenCV
        cv2.rectangle(resized_image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(resized_image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Apply segmentation masks 
    if results_v8[0].masks is not None:
        masks = results_v8[0].masks.data.cpu().numpy()
        for mask in masks:
            # Create a random color mask and apply it to the image
            color_mask = np.random.randint(0, 255, (1, 3), dtype=np.uint8)  # Random color for each mask
            mask_resized = cv2.resize(mask, (resized_image_np.shape[1], resized_image_np.shape[0]))
            mask_applied = (mask_resized[:, :, None] * color_mask).astype(np.uint8)
            resized_image_np = cv2.addWeighted(resized_image_np, 1, mask_applied, 0.5, 0)

    # Convert the final image with bounding boxes and segmentation masks to PIL format
    final_image = Image.fromarray(resized_image_np)
    return final_image

# Streamlit interface
st.title("Object Detection and Segmentation")
st.write("Upload an image to detect objects and apply segmentation masks using YOLOv8.")

# Image uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    st.write("Processing the image...")
    processed_image = process_image(image)

    
    st.image(processed_image, caption='Processed Image with YOLOv8 Detections and Segmentation', use_column_width=True)
    st.success("Image processed successfully!")
