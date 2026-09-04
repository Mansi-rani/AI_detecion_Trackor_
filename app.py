import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os

from detector import ObjectDetector
from tracker import ObjectTracker

from day_night import (
    detect_day_night,
    enhance_night_image
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Vision",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */

    .hero {
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;

        background: linear-gradient(
            135deg,
            #111827,
            #1e3a8a
        );

        color: white;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.85;
    }

    /* Cards */

    .card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 15px;
        color: #6b7280;
    }

    .card-value {
        font-size: 30px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_detector():

    return ObjectDetector(
        "yolo11n.pt"
    )


@st.cache_resource
def load_tracker():

    return ObjectTracker(
        "yolo11n.pt"
    )


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown(
    "# 🎯 AI VISION"
)

st.sidebar.caption(
    "Intelligent Object Detection & Tracking"
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📷 Image Detection",
        "🎥 Video Analysis",
        "📹 Live Camera",
        "📊 Analytics",
        "ℹ️ About"
    ]
)


st.sidebar.markdown("---")


st.sidebar.subheader(
    "⚙️ Detection Settings"
)


detection_mode = st.sidebar.selectbox(
    "AI Mode",
    [
        "Object Detection",
        "Object Tracking"
    ]
)


confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.50,
    0.05
)


# =====================================================
# HERO HEADER
# =====================================================

st.markdown(
    """
    <div class="hero">

        <h1>🎯 AI Vision Dashboard</h1>

        <p>
        Intelligent object detection, tracking and
        adaptive day/night analysis powered by YOLO.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🤖 AI ENGINE
                </div>

                <div class="card-value">
                    YOLO
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🎯 DETECTION
                </div>

                <div class="card-value">
                    READY
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🆔 TRACKING
                </div>

                <div class="card-value">
                    ACTIVE
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🟢 SYSTEM
                </div>

                <div class="card-value">
                    ONLINE
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-title">AI Capabilities</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.info(
            "📷 **Image Detection**\n\n"
            "Detect multiple objects in uploaded images."
        )


    with c2:

        st.info(
            "🎥 **Video Tracking**\n\n"
            "Track objects across video frames."
        )


    with c3:

        st.info(
            "🌞🌙 **Day/Night AI**\n\n"
            "Automatically adapts detection for dark scenes."
        )


    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        **1️⃣ Upload / Capture**

        Provide an image, video or camera feed.

        **2️⃣ Environment Analysis**

        The system checks brightness and identifies
        Day or Night.

        **3️⃣ AI Detection**

        YOLO identifies objects and confidence scores.

        **4️⃣ Tracking**

        Objects can be tracked across video frames.

        **5️⃣ Analytics**

        Detection information is displayed through
        the dashboard.
        """
    )


# =====================================================
# IMAGE DETECTION
# =====================================================

elif page == "📷 Image Detection":

    st.markdown(
        '<div class="section-title">📷 Image Detection</div>',
        unsafe_allow_html=True
    )


    uploaded_image = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_image:

        file_bytes = uploaded_image.read()


        image = cv2.imdecode(
            np.frombuffer(
                file_bytes,
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "Unable to read image."
            )

        else:

            environment, brightness = (
                detect_day_night(image)
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Environment",
                    "🌞 Day"
                    if environment == "Day"
                    else "🌙 Night"
                )


            with col2:

                st.metric(
                    "Brightness",
                    f"{brightness:.1f}"
                )


            with col3:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.0f}%"
                )


            processing_image = image.copy()


            if environment == "Night":

                processing_image = (
                    enhance_night_image(
                        image
                    )
                )

                st.info(
                    "🌙 Night mode active — "
                    "image enhancement enabled."
                )

            else:

                st.success(
                    "🌞 Day mode active."
                )


            with st.spinner(
                "🤖 AI is analyzing image..."
            ):

                if detection_mode == "Object Detection":

                    detector = load_detector()

                    result = detector.detect(
                        processing_image,
                        confidence
                    )

                else:

                    tracker = load_tracker()

                    result = tracker.track(
                        processing_image,
                        confidence
                    )


                output = result.plot()


            st.success(
                "✅ Detection completed!"
            )


            st.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                caption="AI Detection Result",
                width="stretch"
            )


# =====================================================
# VIDEO ANALYSIS
# =====================================================

elif page == "🎥 Video Analysis":

    st.markdown(
        '<div class="section-title">🎥 Video Analysis</div>',
        unsafe_allow_html=True
    )


    uploaded_video = st.file_uploader(
        "Upload video",
        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )


    if uploaded_video:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )


        temp_file.write(
            uploaded_video.read()
        )

        temp_file.close()


        cap = cv2.VideoCapture(
            temp_file.name
        )


        if not cap.isOpened():

            st.error(
                "Unable to open video."
            )

        else:

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )


            fps = cap.get(
                cv2.CAP_PROP_FPS
            )


            if fps <= 0:
                fps = 30


            st.info(
                f"🎬 Frames: {total_frames} | "
                f"FPS: {fps:.1f}"
            )


            if detection_mode == "Object Detection":

                detector = load_detector()

            else:

                tracker = load_tracker()


            video_placeholder = st.empty()

            progress = st.progress(0)

            frame_count = 0

            start_time = time.time()


            with st.spinner(
                "🤖 AI is analyzing video..."
            ):

                while True:

                    success, frame = cap.read()


                    if not success:
                        break


                    frame_count += 1


                    environment, brightness = (
                        detect_day_night(
                            frame
                        )
                    )


                    processing_frame = frame.copy()


                    if environment == "Night":

                        processing_frame = (
                            enhance_night_image(
                                frame
                            )
                        )


                    if detection_mode == "Object Detection":

                        result = detector.detect(
                            processing_frame,
                            confidence
                        )

                    else:

                        result = tracker.track(
                            processing_frame,
                            confidence
                        )


                    output = result.plot()


                    cv2.putText(
                        output,
                        f"Environment: {environment}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )


                    cv2.putText(
                        output,
                        f"Brightness: {brightness:.1f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )


                    video_placeholder.image(
                        cv2.cvtColor(
                            output,
                            cv2.COLOR_BGR2RGB
                        ),
                        channels="RGB",
                        width="stretch"
                    )


                    if total_frames > 0:

                        progress.progress(
                            min(
                                int(
                                    (
                                        frame_count /
                                        total_frames
                                    ) * 100
                                ),
                                100
                            )
                        )


            cap.release()


            try:
                os.remove(
                    temp_file.name
                )
            except:
                pass


            elapsed = (
                time.time() - start_time
            )


            st.success(
                f"✅ Analysis completed! "
                f"{frame_count} frames processed "
                f"in {elapsed:.1f} seconds."
            )


# =====================================================
# LIVE CAMERA
# =====================================================

elif page == "📹 Live Camera":

    st.markdown(
        '<div class="section-title">📹 Live Camera</div>',
        unsafe_allow_html=True
    )


    st.info(
        "Capture an image using your webcam "
        "and run AI detection."
    )


    camera_image = st.camera_input(
        "Capture an image from your camera"
    )


    if camera_image:

        file_bytes = camera_image.getvalue()


        frame = cv2.imdecode(
            np.frombuffer(
                file_bytes,
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )


        if frame is None:

            st.error(
                "Unable to read camera image."
            )

        else:

            environment, brightness = (
                detect_day_night(
                    frame
                )
            )


            processing_frame = frame.copy()


            if environment == "Night":

                processing_frame = (
                    enhance_night_image(
                        frame
                    )
                )


            detector = load_detector()


            result = detector.detect(
                processing_frame,
                confidence
            )


            output = result.plot()


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Environment",
                    environment
                )


            with col2:

                st.metric(
                    "Brightness",
                    f"{brightness:.1f}"
                )


            st.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                caption=(
                    f"Camera Result — "
                    f"{environment} Mode"
                ),
                width="stretch"
            )


# =====================================================
# ANALYTICS
# =====================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-title">📊 AI Analytics</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Detection Engine",
            "YOLO"
        )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.0f}%"
        )


    with col3:

        st.metric(
            "Tracking",
            "Enabled"
            if detection_mode == "Object Tracking"
            else "Disabled"
        )


    st.markdown("---")


    st.subheader(
        "📈 Detection System Information"
    )


    st.write(
        """
        The AI Vision system uses YOLO for object
        detection and tracking.

        The system also analyzes image brightness
        to determine whether the environment is
        Day or Night.

        For dark scenes, CLAHE-based enhancement
        is applied before object detection.
        """
    )


    st.markdown("### System Configuration")


    st.write(
        {
            "AI Model": "YOLO11 Nano",
            "Detection Mode": detection_mode,
            "Confidence Threshold": confidence,
            "Day/Night Detection": "Enabled",
            "Night Enhancement": "Enabled",
            "Image Detection": "Enabled",
            "Video Analysis": "Enabled",
            "Camera Input": "Enabled"
        }
    )


# =====================================================
# ABOUT
# =====================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">ℹ️ About Project</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        ## 🎯 AI Object Detection & Tracking System

        This project uses artificial intelligence to
        detect and track objects in images and videos.

        ### 🧠 Technologies

        - Python
        - YOLO / Ultralytics
        - OpenCV
        - Streamlit
        - NumPy

        ### ⭐ Main Features

        - Object Detection
        - Object Tracking
        - Day/Night Detection
        - Night Image Enhancement
        - Image Analysis
        - Video Analysis
        - Camera Input
        - Confidence Control
        - Interactive Dashboard

        ### 🚀 Future Improvements

        - Object Counting
        - Tracking IDs
        - Real-time FPS
        - Advanced Analytics
        - Zone Monitoring
        - Detection History
        - Result Download
        """
    )
