import streamlit as st
import torch
import numpy as np
import os
import cv2
import tempfile
import pandas as pd
import altair as alt
import re
from transformers import AutoProcessor, AutoModelForVideoClassification
import matplotlib.pyplot as plt
import gdown

# Set page configuration
st.set_page_config(layout="wide", page_title="Action Recognition")

# Sidebar
st.sidebar.write("## Upload and Process Video 🎥")
uploaded_file = st.sidebar.file_uploader("Upload a video file:", type=["mp4", "avi", "mov"])

# Sidebar Information
with st.sidebar.expander("ℹ Video Guidelines"):
    st.write("""
    - Supported formats: MP4, AVI, MOV
    - Ensure the video contains clear actions for better predictions
    """)



@st.cache_resource
def load_model():
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    processor = AutoProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    return model, processor

# Function to extract frames from a video
def extract_frames_from_video(video_path, output_folder, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    frame_count = 0
    saved_frames = 0
    while cap.isOpened() and saved_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_frames + 1:04d}.jpg")
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        frame_count += 1
    cap.release()

# Main Layout
st.write("## Action Recognition App")
st.write("Upload a video to predict the action using a pre-trained model.")

# Introduction
st.write("""
This app allows you to upload a video, converts it into frames, and predicts the action using a pre-trained model.
We use **TimeSformer**, a state-of-the-art video transformer model, which processes video frames as a sequence of images and captures temporal relationships to predict actions effectively.
Experience seamless action recognition with visualizations and confidence scores.
""")

col1, col2 = st.columns(2)

# Load model and processor
model, processor = load_model()

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded video
        col1.write("### Uploaded Video")
        col1.video(video_path)

        # Extract frames from the video
        st.info("Extracting frames from the video...")
        extract_frames_from_video(video_path, temp_dir, num_frames=8)
        folder_path = temp_dir

        # Process the extracted frames
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])[:8]
        frames = []

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            frame = cv2.imread(img_path)
            frames.append(frame)

        if len(frames) < 8:
            st.warning("The video must contain enough frames to extract 8 frames.")
        else:
            inputs = processor([frames], return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_prob, top_index = torch.max(probs, dim=-1)

            # Display the single top prediction
            col2.write("### Predicted Action")
            action_label = model.config.id2label[top_index.item()]
            confidence = top_prob.item() * 100
            col2.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
                    <h2 style="font-size: 24px; color: #4CAF50;">{action_label}</h2>
                    <p style="font-size: 16px; color: #777;">Confidence: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


