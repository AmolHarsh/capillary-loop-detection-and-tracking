# Capillary Loop Detection and Tracking

This is a Streamlit-powered application designed as a prototype to demonstrate the capabilities of utilizing the latest object detection tools in capillary analysis. By leveraging the YOLO (You Only Look Once) object detection model, it provides a user-friendly interface for analyzing capillary structures in medical images and videos.

**Note:** This project is still a work in progress and is part of a research project at the University of California, San Diego (UCSD).

## Project Purpose

This project serves as a prototype to showcase how advanced object detection models like YOLO can be applied to the analysis of capillary loops in medical imaging. It is intended for research and educational purposes to explore the potential of AI in medical diagnostics.

## Features

- **Image and Video Processing**: Upload medical images or videos for automated capillary loop detection.
- **Real-time Object Tracking**: Track capillary loops across video frames.
- **Downloadable Results**: Save processed videos with detected loops highlighted.
- **Adjustable Thresholds**: Customize confidence and IOU thresholds for optimal detection and tracking.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AmolHarsh/capillary-loop-detection-and-tracking.git
    cd capillary-loop-detection-and-tracking
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run CapWeb.py
    ```

### How to Use

1. **Upload an Image or Video**:
    - Use the interface to upload a medical image or video file for capillary loop detection.
    
2. **Adjust Settings** (Optional):
    - Modify confidence and IOU thresholds to fine-tune detection accuracy.

3. **View and Download Results**:
    - For images, view the detected capillary loops directly in the app.
    - For videos, download the processed video with tracking annotations.

### Project Structure

- `CapWeb.py`: The main Streamlit application script.
- `requirements.txt`: List of Python dependencies required to run the app.
- `README.md`: Project overview and usage instructions (this file).

### Acknowledgments

- **Ultralytics YOLO**: For the robust object detection model.
- **Streamlit**: For providing a simple and effective framework for web apps.

---

**Disclaimer:** This project is a prototype and is intended to demonstrate the capabilities of using the latest object detection tools in capillary analysis. It is not intended for clinical use and should not be considered as a substitute for professional medical advice, diagnosis, or treatment.
