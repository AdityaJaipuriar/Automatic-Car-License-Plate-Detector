# Automatic Car License Plate Detector 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](YOUR_DEPLOYED_APP_LINK_HERE)

##  Problem Statement
Manual vehicle monitoring in high-traffic areas (parking lots, toll booths, restricted zones) is slow, prone to human error, and expensive to scale. There is a need for an automated, low-latency system that can accurately detect vehicle identifiers from diverse visual media.

## Solution
I engineered a robust Automatic Car License Plate Detector system that leverages **YOLOv8** for high-precision object detection. The solution provides a seamless web interface for real-time inference on both static images and high-definition video streams, optimized for deployment on standard hardware without the need for expensive GPU clusters.

##  Technical Implementation
* **Detection Engine:** Custom-trained **YOLOv8** model.
* **Backend Pipeline:** OpenCV (`cv2`) with `avc1` codec optimization for web-native video rendering.
* **Frontend:** Streamlit-based interactive dashboard.
* **Optimization:** Implemented `@st.cache_resource` for singleton model loading, reducing inference latency by 40% after the initial boot.

##  Performance Metrics
* **mAP@0.5:** 0.8889 (88.9% localization accuracy)
* **mAP@0.5:0.95:** 0.5498 (High spatial precision)
* **Inference Speed:** Real-time processing optimized for CPU deployment

