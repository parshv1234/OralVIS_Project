import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

MODEL_PATH = 'runs/detect/oralvis_yolov8m_run1/weights/best.pt'

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model from the specified path."""
    return YOLO(model_path)

model = load_model(MODEL_PATH)

# --- POST-PROCESSING LOGIC ---
def post_process_predictions(boxes, img_width, img_height):
    """Applies anatomical logic to sort and organize tooth detections."""
    quadrants = {1: [], 2: [], 3: [], 4: []}
    if not boxes:
        return quadrants

    all_x_centers = [b['center_x'] for b in boxes]
    all_y_centers = [b['center_y'] for b in boxes]
    x_midline = np.mean(all_x_centers)
    y_midline = np.mean(all_y_centers)

    for box in boxes:
        if box['center_y'] < y_midline:  # Upper arch
            if box['center_x'] > x_midline: # Quadrant 1
                quadrants[1].append(box)
            else: # Quadrant 2
                quadrants[2].append(box)
        else:  # Lower arch
            if box['center_x'] < x_midline: # Quadrant 3
                quadrants[3].append(box)
            else: # Quadrant 4
                quadrants[4].append(box)

    quadrants[1] = sorted(quadrants[1], key=lambda b: b['center_x'])
    quadrants[4] = sorted(quadrants[4], key=lambda b: b['center_x'])
    quadrants[2] = sorted(quadrants[2], key=lambda b: b['center_x'], reverse=True)
    quadrants[3] = sorted(quadrants[3], key=lambda b: b['center_x'], reverse=True)
    
    return quadrants

def draw_results(image, sorted_quadrants):
    """Draws the sorted bounding boxes on the image, color-coded by quadrant."""
    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)} # BGR
    
    for quad_num, teeth in sorted_quadrants.items():
        for i, tooth in enumerate(teeth):
            x1, y1, x2, y2 = tooth['coords']
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[quad_num], 2)
            
            # MODIFIED: Add confidence score to the label text
            confidence_percent = int(tooth['confidence'] * 100)
            label_text = f"Q{quad_num}-{i+1} {tooth['label'].split('(')[0]} {confidence_percent}%"
            
            cv2.putText(image, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[quad_num], 2)
    return image

def generate_text_report(sorted_quadrants):
    """Generates a formatted string report of the detected teeth."""
    report = "OralVis Analysis Report\n"
    report += "="*25 + "\n\n"
    for i in range(1, 5):
        report += f"--- Quadrant {i} ---\n"
        if sorted_quadrants[i]:
            for tooth in sorted_quadrants[i]:
                # MODIFIED: Add confidence score to the text report
                confidence_percent = int(tooth['confidence'] * 100)
                report += f"- {tooth['label']} ({confidence_percent}% confidence)\n"
        else:
            report += "No teeth detected in this quadrant.\n"
        report += "\n"
    return report

# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🦷 OralVis: AI Dental X-Ray Analysis")
st.write("Upload a panoramic dental X-ray to detect, classify, and sort teeth using a trained YOLOv8 model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    H, W, _ = cv2_img.shape

    st.sidebar.header("Original Image")
    st.sidebar.image(cv2_img, channels="BGR", use_column_width=True)

    results = model(cv2_img)[0]
    
    boxes_for_pp = []
    for box in results.boxes:
        class_id = int(box.cls)
        label = model.names[class_id]
        confidence = float(box.conf) # MODIFIED: Extract confidence score
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].cpu().numpy()]
        
        # MODIFIED: Add confidence to the dictionary
        boxes_for_pp.append({
            "coords": (x1, y1, x2, y2),
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "label": label,
            "confidence": confidence 
        })

    sorted_quadrants = post_process_predictions(boxes_for_pp, W, H)

    output_image = draw_results(cv2_img.copy(), sorted_quadrants)
    st.header("Analysis Results")
    st.image(output_image, channels="BGR", caption="Post-processed detections with anatomical sorting and confidence scores.")

    st.header("Detected Teeth (Sorted)")
    cols = st.columns(4)
    for i in range(1, 5):
        with cols[i-1]:
            st.subheader(f"Quadrant {i}")
            if sorted_quadrants[i]:
                for tooth in sorted_quadrants[i]:
                    # MODIFIED: Display confidence score in the text list
                    confidence_percent = int(tooth['confidence'] * 100)
                    st.write(f"- {tooth['label']} ({confidence_percent}%)")
            else:
                st.write("No teeth detected.")

    st.header("Download Results")
    col1, col2 = st.columns(2)

    with col1:
        is_success, buffer = cv2.imencode(".png", output_image)
        byte_im = buffer.tobytes()
        st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name=f"processed_{uploaded_file.name}.png",
            mime="image/png"
        )
    
    with col2:
        text_report = generate_text_report(sorted_quadrants)
        st.download_button(
            label="Download Text Report",
            data=text_report,
            file_name="analysis_report.txt",
            mime="text/plain"
        )