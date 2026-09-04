import streamlit as st
import cv2
import numpy as np
import tempfile
import time

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
# LOAD AI MODELS
# =====================================================

@st.cache_resource
def load_detector():
    return ObjectDetector()


@st.cache_resource
def load_tracker():
    return ObjectTracker()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎯 AI VISION")

st.sidebar.write(
    "Intelligent Object Detection & Tracking"
)

st.sidebar.divider()

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

st.sidebar.divider()

st.sidebar.subheader("⚙️ Detection Settings")

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
# MAIN HEADER
# =====================================================

st.title("🎯 AI Vision Dashboard")

st.write(
    "Intelligent object detection, tracking and "
    "adaptive day/night analysis powered by YOLO."
)

st.divider()


# =====================================================
# DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🤖 AI ENGINE",
            "YOLO"
        )

    with col2:
        st.metric(
            "🎯 DETECTION",
            "READY"
        )

    with col3:
        st.metric(
            "🆔 TRACKING",
            "ACTIVE"
        )

    with col4:
        st.metric(
            "🟢 SYSTEM",
            "ONLINE"
        )

    st.header("AI Capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "📷 Image Detection\n\n"
            "Detect multiple objects in uploaded images."
        )

    with c2:
        st.info(
            "🎥 Video Tracking\n\n"
            "Track objects across video frames."
        )

    with c3:
        st.info(
            "🌞🌙 Day/Night AI\n\n"
            "Automatically adapts detection for dark scenes."
        )

    st.header("How It Works")

    st.write("1️⃣ Upload or capture an image/video.")

    st.write(
        "2️⃣ The system analyzes brightness "
        "and identifies Day or Night."
    )

    st.write(
        "3️⃣ YOLO identifies objects and "
        "confidence scores."
    )

    st.write(
        "4️⃣ Objects can be tracked across "
        "video frames."
    )

    st.write(
        "5️⃣ Detection information is displayed "
        "through the dashboard."
    )


# =====================================================
# IMAGE DETECTION
# =====================================================

elif page == "📷 Image Detection":

    st.header("📷 Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
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

            st.error("Unable to read image.")

        else:

            environment, brightness = detect_day_night(
                image
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if environment == "Day":
                    st.metric(
                        "Environment",
                        "🌞 Day"
                    )
                else:
                    st.metric(
                        "Environment",
                        "🌙 Night"
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

            processing_image = image

            if environment == "Night":

                processing_image = enhance_night_image(
                    image
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

    st.header("🎥 Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
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
                        detect_day_night(frame)
                    )

                    processing_frame = frame

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
                                frame_count /
                                total_frames,
                                1.0
                            )
                        )

            cap.release()

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

    st.header("📹 Live Camera")

    st.info(
        "Capture an image using your webcam."
    )

    camera_image = st.camera_input(
        "Take a picture"
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

        environment, brightness = (
            detect_day_night(frame)
        )

        if environment == "Night":

            frame = enhance_night_image(
                frame
            )

        detector = load_detector()

        result = detector.detect(
            frame,
            confidence
        )

        output = result.plot()

        st.image(
            cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"Camera Result — {environment} Mode",
            width="stretch"
        )


# =====================================================
# ANALYTICS
# =====================================================

elif page == "📊 Analytics":

    st.header("📊 AI Analytics")

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

    st.divider()

    st.subheader(
        "📈 Detection System Information"
    )

    st.write(
        "The analytics section can be extended "
        "to display object counts, tracking IDs, "
        "FPS, detection history and day/night statistics."
    )


# =====================================================
# ABOUT
# =====================================================

elif page == "ℹ️ About":

    st.header("ℹ️ About Project")

    st.subheader(
        "🎯 AI Object Detection & Tracking System"
    )

    st.write(
        "This project uses artificial intelligence "
        "to detect and track objects in images and videos."
    )

    st.subheader("🧠 Technologies")

    st.write(
        "- Python\n"
        "- YOLO / Ultralytics\n"
        "- OpenCV\n"
        "- Streamlit\n"
        "- NumPy\n"
        "- Pandas"
    )

    st.subheader("⭐ Main Features")

    st.write(
        "- Object Detection\n"
        "- Object Tracking\n"
        "- Day/Night Detection\n"
        "- Night Image Enhancement\n"
        "- Image Analysis\n"
        "- Video Analysis\n"
        "- Camera Input\n"
        "- Confidence Control\n"
        "- Interactive Dashboard"
    )

    st.subheader("🚀 Future Improvements")

    st.write(
        "- Object Counting\n"
        "- Tracking IDs\n"
        "- Real-time FPS\n"
        "- Advanced Analytics\n"
        "- Zone Monitoring\n"
        "- Detection History\n"
        "- Result Download"
    )