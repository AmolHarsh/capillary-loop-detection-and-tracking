import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import time
import tempfile
import os
import torch

# Check if MPS is available and select the device accordingly
device = torch.device("cpu")

# Load the model using Ultralytics
model = YOLO('models/bestNano.pt').to(device)

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_area = max(0, x2_min - x1_max + 1) * max(0, y2_min - y1_max + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Function to filter overlapping boxes and keep the smallest
def filter_boxes(boxes, confidences, iou_threshold=0.5):
    keep_boxes = []
    keep_confidences = []

    while len(boxes) > 0:
        smallest_box_idx = np.argmin([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
        smallest_box = boxes[smallest_box_idx]
        smallest_confidence = confidences[smallest_box_idx]

        keep = True
        for i in range(len(keep_boxes)):
            iou = calculate_iou(smallest_box, keep_boxes[i])
            if iou > iou_threshold:
                keep = False
                break

        if keep:
            keep_boxes.append(smallest_box)
            keep_confidences.append(smallest_confidence)

        # Remove the smallest box from the list
        boxes = np.delete(boxes, smallest_box_idx, axis=0)
        confidences = np.delete(confidences, smallest_box_idx, axis=0)

    return np.array(keep_boxes), np.array(keep_confidences)

# Define the prediction function for images
def predict(image):
    start_time = time.time()  # Start timing
    results = model(image)  # Perform inference
    inference_time = time.time() - start_time  # Calculate inference time
    return results, inference_time

# Function to trim video to the first 30 seconds
def trim_video(video_path, max_duration=5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_duration * fps))

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return output_video_path

def draw_boxes_with_confidence(image, results):
    # Check if the input is a PIL image or a numpy array
    if isinstance(image, np.ndarray):
        image_np = image  # Already a numpy array
    else:
        image_np = np.array(image.convert('RGB'))  # Convert PIL image to numpy array

    height, width = image_np.shape[:2]
    total_capillaries = 0  # Initialize capillary count

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get the bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            
            # Filter overlapping boxes
            boxes, confidences = filter_boxes(boxes, confidences)
            
            total_capillaries += len(boxes)
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)

                # Draw the bounding box
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # Draw the confidence value
                cv2.putText(image_np, f'{conf:.2f}', (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
    
    return image_np, total_capillaries


# Function to process and annotate video frames with progress tracking
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)  # Initialize the progress bar
    progress_text = st.empty()  # Placeholder for the progress percentage text

    total_capillaries = 0  # Initialize capillary count for the entire video
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the YOLO model with tracking on the frame
        results = model.track(frame, conf=0.08, iou=0.5)

        # Annotate the frame with tracking results
        annotated_frame, capillaries_in_frame = draw_boxes_with_confidence(frame, results)
        
        # Update the total number of capillaries detected
        total_capillaries = max(total_capillaries, capillaries_in_frame)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Update the progress bar and text
        current_frame += 1
        progress = int((current_frame / total_frames) * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Processing: {progress}%")

    # Release everything
    cap.release()
    out.release()

    return output_video_path, total_capillaries

# Streamlit app
st.title('Capillary Loop Detection and Tracking')

# Upload options for image or video
upload_type = st.radio("Select upload type:", ("Image", "Video"))

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose a capillary image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict button
        if st.button('Predict'):
            results, inference_time = predict(image)
            
            # Display the inference time
            st.write(f"Inference Time: {inference_time:.2f} seconds")

            # Process and display the results
            for result in results:
                st.write('Predicted Results:')
                if result.boxes is not None:
                    # Annotate image with bounding boxes and confidence values
                    annotated_image, total_capillaries = draw_boxes_with_confidence(image, [result])

                    # Display the number of capillaries detected
                    st.write(f"Number of capillaries detected: {total_capillaries}")

                    # Convert the annotated numpy image back to PIL format for display
                    annotated_image = Image.fromarray(annotated_image)

                    # Display the annotated image
                    st.image(annotated_image, caption='Annotated Image', use_column_width=True)
                else:
                    st.write("No bounding boxes detected.")

elif upload_type == "Video":
    uploaded_file = st.file_uploader("Choose a capillary video...", type=["mp4", "avi", "av"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Predict button
        if st.button('Predict'):
            # Trim the video to the first 30 seconds
            trimmed_video_path = trim_video(tfile.name)

            # Process the trimmed video for tracking with progress bar
            output_video_path, total_capillaries = process_video(trimmed_video_path)
            
            # Display the number of capillaries detected in the entire video
            st.write(f"Maximum number of capillaries detected in a frame: {total_capillaries}")
            
            
            # Ensure the video is fully written before trying to download it
            if os.path.exists(output_video_path):
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
