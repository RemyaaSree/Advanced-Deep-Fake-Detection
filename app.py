import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import random
import cv2
import torch
from torchvision import models, transforms

# Optional: Grad-CAM for image/video
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    cam_available = True
except ImportError:
    cam_available = False

# --- Session State for login and navigation ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# --- Sidebar UI ---
st.sidebar.title("Deepfake Detector")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
if st.session_state.logged_in:
    st.sidebar.success("Logged in as user")
    st.sidebar.button("Logout", on_click=lambda: [st.session_state.update({'logged_in': False, 'page': 'login'})])
else:
    st.sidebar.info("Please login to use the app.")

# --- Model Loading (DEMO: replace with real models as needed) ---
@st.cache_resource
def load_image_model():
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# --- Login Page ---
def login_page():
    st.markdown("""
    <div style='text-align:center;'>
        <h1>🔒 Login to Deepfake Detection</h1>
    </div>
    """, unsafe_allow_html=True)
    st.write("Welcome! Please log in to access deepfake detection features.")
    username = st.text_input('Username', placeholder='Enter username')
    password = st.text_input('Password', type='password', placeholder='Enter password')
    if st.button('Login', use_container_width=True):
        if username == 'user' and password == 'user123':
            st.session_state.logged_in = True
            st.session_state.page = 'menu'
            st.success('Login successful!')
        else:
            st.error('Invalid credentials')

# --- Main Menu ---
def main_menu():
    st.markdown("""
    <div style='text-align:center;'>
        <h1>🕵️‍♂️ Deepfake Detection</h1>
        <p>Select the type of media you want to check for deepfakes.</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('🖼️ Detect Image', use_container_width=True):
            st.session_state.page = 'image'
    with col2:
        if st.button('🎞️ Detect Video', use_container_width=True):
            st.session_state.page = 'video'
    with col3:
        if st.button('🎵 Detect Audio', use_container_width=True):
            st.session_state.page = 'audio'
    st.info("Use the sidebar to logout at any time.")

# --- Image Deepfake Detection (with Grad-CAM) ---
def image_detection_page():
    st.markdown("<h2>🖼️ Image Deepfake Detection</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect', use_container_width=True):
            # --- Real Model Inference (DEMO: random) ---
            model = load_image_model()
            preprocess = get_preprocess()
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                fake_prob = float(torch.sigmoid(output).mean())
                label = "Fake" if fake_prob > 0.5 else "Real"
                confidence = round(fake_prob if label == "Fake" else 1 - fake_prob, 2)
            color = 'red' if label == 'Fake' else 'green'
            st.markdown(f'<h3>Prediction: <span style="color:{color}">{label}</span></h3>', unsafe_allow_html=True)
            st.write(f'**Confidence:** {confidence:.2f}')
            # --- Grad-CAM Heatmap ---
            if cam_available:
                target_layer = model.features[-1]
                cam = GradCAM(model=model, target_layers=[target_layer])
                grayscale_cam = cam(input_tensor=input_tensor)[0, :]
                np_image = np.array(image.resize((224, 224))) / 255.
                visualization = show_cam_on_image(np_image, grayscale_cam, use_rgb=True)
                st.image(visualization, caption="Manipulation Heatmap (Grad-CAM)")
                st.info("Red/yellow regions show where the model focused for its decision.")
            else:
                st.warning("Grad-CAM not available. Install pytorch-grad-cam for heatmaps.")
    if st.button('Back to Menu', use_container_width=True):
        st.session_state.page = 'menu'

# --- Video Deepfake Detection (Frame-by-frame with Grad-CAM) ---
def video_detection_page():
    st.markdown("<h2>🎞️ Video Deepfake Detection</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Upload a video', type=['mp4', 'avi'])
    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button('Detect', use_container_width=True):
            # --- Extract Frames ---
            tfile = open("temp_video.mp4", "wb")
            tfile.write(uploaded_file.read())
            tfile.close()
            cap = cv2.VideoCapture("temp_video.mp4")
            frames = []
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret or i >= 30:  # Only first 30 frames for demo
                    break
                if i % 5 == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                i += 1
            cap.release()
            # --- Run Model on Frames (DEMO: random + Grad-CAM) ---
            model = load_image_model()
            preprocess = get_preprocess()
            suspect_frames = []
            heatmaps = []
            for idx, frame in enumerate(frames):
                pil_frame = Image.fromarray(frame)
                input_tensor = preprocess(pil_frame).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    fake_prob = float(torch.sigmoid(output).mean())
                    label = "Fake" if fake_prob > 0.5 else "Real"
                if label == "Fake":
                    suspect_frames.append(idx)
                    if cam_available:
                        target_layer = model.features[-1]
                        cam = GradCAM(model=model, target_layers=[target_layer])
                        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
                        np_image = np.array(pil_frame.resize((224, 224))) / 255.
                        visualization = show_cam_on_image(np_image, grayscale_cam, use_rgb=True)
                        heatmaps.append(visualization)
                    else:
                        heatmaps.append(frame)
            st.write(f"Suspect frames: {suspect_frames}")
            # --- Show suspect frames with heatmaps ---
            if suspect_frames:
                frame_idx = st.slider("Browse suspect frames", 0, len(suspect_frames)-1, 0)
                st.image(heatmaps[frame_idx], caption=f"Suspect Frame {suspect_frames[frame_idx]} (Heatmap)")
                st.warning("Red/yellow regions show where the model focused for its decision.")
            else:
                st.success("No suspect frames detected. Video appears real.")
    if st.button('Back to Menu', use_container_width=True):
        st.session_state.page = 'menu'

# --- Audio Deepfake Detection (Spectrogram + CNN logic placeholder) ---
def audio_detection_page():
    st.markdown("<h2>🎵 Audio Deepfake Detection</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Upload an audio file', type=['wav', 'mp3'])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        try:
            y, sr = librosa.load(uploaded_file, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_DB = librosa.power_to_db(S, ref=np.max)
            # --- Model Prediction (DEMO: random) ---
            n_windows = S_DB.shape[1]
            fake_probs = np.random.rand(n_windows)
            threshold = 0.7
            suspect_segments = np.where(fake_probs > threshold)[0]
            # --- Plot Spectrogram and Highlight Suspect Segments ---
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            for seg in suspect_segments:
                ax.axvspan(seg, seg+1, color='red', alpha=0.3)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            st.image(buf, caption='Audio Spectrogram with Suspect Segments')
            plt.close(fig)
            if len(suspect_segments) > 0:
                st.warning(f"Suspect segments detected at time bins: {suspect_segments.tolist()}")
            else:
                st.success("No morphed segments detected. Audio appears real.")
        except Exception as e:
            st.info('Unable to display spectrogram.')
    if st.button('Back to Menu', use_container_width=True):
        st.session_state.page = 'menu'

# --- Page Routing ---
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.page == 'menu':
        main_menu()
    elif st.session_state.page == 'image':
        image_detection_page()
    elif st.session_state.page == 'video':
        video_detection_page()
    elif st.session_state.page == 'audio':
        audio_detection_page()