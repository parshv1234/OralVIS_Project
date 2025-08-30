import cv2
import numpy as np
from ultralytics import YOLO
import os

MODEL_PATH = 'runs/detect/oralvis_yolov8m_run1/weights/best.pt'

IMAGE_PATH = 'OralVis_Dataset/images/test/cate8-00116_jpg.rf.8b5652f7babcf3d99f5b2f52785d963b.jpg' 

output_directory = 'postprocessing'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: '{output_directory}'")

def get_predictions(model, image_path):
    """Runs the YOLO model on an image and returns the detected boxes."""
    img = cv2.imread(image_path)
    H, W, _ = img.shape
    
    results = model(image_path)[0]
    
    boxes = []
    for box in results.boxes:
        class_id = int(box.cls)
        label = model.names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
        
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        
        boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": label, "confidence": confidence
        })
    return boxes, img

def post_process_predictions(boxes, img_width, img_height):
    """Applies anatomical logic to sort and organize tooth detections."""
    
    quadrants = {1: [], 2: [], 3: [], 4: []}
    
    if not boxes:
        return quadrants

    all_x_centers = [(b['x1'] + b['x2']) / 2 for b in boxes]
    all_y_centers = [(b['y1'] + b['y2']) / 2 for b in boxes]
    x_midline = np.mean(all_x_centers)
    y_midline = np.mean(all_y_centers)

    for box in boxes:
        center_x = (box['x1'] + box['x2']) / 2
        center_y = (box['y1'] + box['y2']) / 2
        
        if center_y < y_midline:  # Upper arch
            if center_x > x_midline: # Quadrant 1 (Patient's Upper Right)
                quadrants[1].append(box)
            else: # Quadrant 2 (Patient's Upper Left)
                quadrants[2].append(box)
        else:  # Lower arch
            if center_x < x_midline: # Quadrant 3 (Patient's Lower Left)
                quadrants[3].append(box)
            else: # Quadrant 4 (Patient's Lower Right)
                quadrants[4].append(box)

    # --- Sort Teeth Horizontally ---
    # Patient's right side (Quadrants 1 & 4) are sorted left-to-right on the image
    quadrants[1] = sorted(quadrants[1], key=lambda b: (b['x1'] + b['x2']) / 2)
    quadrants[4] = sorted(quadrants[4], key=lambda b: (b['x1'] + b['x2']) / 2)
    
    # Patient's left side (Quadrants 2 & 3) are sorted right-to-left on the image
    quadrants[2] = sorted(quadrants[2], key=lambda b: (b['x1'] + b['x2']) / 2, reverse=True)
    quadrants[3] = sorted(quadrants[3], key=lambda b: (b['x1'] + b['x2']) / 2, reverse=True)
    
    return quadrants

def draw_results(image, sorted_quadrants):
    """Draws the sorted bounding boxes on the image, color-coded by quadrant."""
    # Colors for each quadrant (BGR format)
    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}
    
    for quad_num, teeth in sorted_quadrants.items():
        print(f"\n--- Quadrant {quad_num} (Sorted) ---")
        for i, tooth in enumerate(teeth):
            # missing teeth would happen here.
            print(f"  {i+1}. {tooth['label']} (Confidence: {tooth['confidence']:.2f})")
            
            # Draw bounding box
            cv2.rectangle(image, (tooth['x1'], tooth['y1']), (tooth['x2'], tooth['y2']), colors[quad_num], 2)
            
            # Draw label
            label_text = f"Q{quad_num}-{i+1} {tooth['label'].split('(')[0]}"
            cv2.putText(image, label_text, (tooth['x1'], tooth['y1'] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[quad_num], 2)

    return image


def main():
    """Main function to run the prediction and post-processing pipeline."""
    model = YOLO(MODEL_PATH)

    raw_boxes, original_image = get_predictions(model, IMAGE_PATH)
    
    H, W, _ = original_image.shape
    sorted_quadrants = post_process_predictions(raw_boxes, W, H)

    output_image = draw_results(original_image.copy(), sorted_quadrants)

    output_filename = 'postprocessing/post_processed_result(1).jpg'
    cv2.imwrite(output_filename, output_image)
    print(f"\n Post-processed result saved to '{output_filename}'")
    
    # To display the image in a window (optional, press 'q' to close)
    cv2.imshow('Post-processed Result', output_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()