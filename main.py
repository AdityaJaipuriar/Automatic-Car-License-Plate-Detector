import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# title of the Streamlit app
st.title("YOLO Image and Video Processing")

# Allowing users to upload images or videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

os.makedirs("temp", exist_ok=True)

# Cache the YOLO model so it only loads once, making the app much faster!
@st.cache_resource
def load_model():
    try:
        return YOLO('best_license_plate_model.pt')  
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

def predict_and_save_image(path_test_car, output_image_path):
    try:
        results = model.predict(path_test_car, device='cpu')
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Change codec to avc1 for web browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Adding a progress bar so the user knows the app hasn't frozen
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu', verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            out.write(frame)
            
            # Updating progress bar
            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))
                
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def process_media(input_path, output_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        # Force the output video to be a .mp4 file so Streamlit can play it
        base_name = os.path.splitext(output_path)[0]
        forced_mp4_output = base_name + ".mp4"
        return predict_and_plot_video(input_path, forced_mp4_output)
        
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

if uploaded_file is not None:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")
    
    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Processing... Please wait.")
        
        result_path = process_media(input_path, output_path)
        
        if result_path:
            if result_path.endswith('.mp4'):
                video_file = open(result_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.image(result_path)
    except Exception as e:
        st.error(f"Error uploading or processing file: {e}")