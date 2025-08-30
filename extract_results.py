import os
import shutil


TRAIN_RUN_FOLDER = 'runs/detect/oralvis_yolov8m_run1' 
TEST_RUN_FOLDER = 'runs/detect/val'

DESTINATION_FOLDER = 'Final_Submission_Files'
# ---------------------

def extract_submission_files():
    print("--- Starting Final File Extraction ---")

    if not os.path.exists(TRAIN_RUN_FOLDER):
        print(f"\n ERROR: Training folder not found at '{TRAIN_RUN_FOLDER}'.")
        print("Please double-check the 'TRAIN_RUN_FOLDER' variable.")
        return
    if not os.path.exists(TEST_RUN_FOLDER):
        print(f"\n ERROR: Test evaluation folder not found at '{TEST_RUN_FOLDER}'.")
        print("Please double-check the 'TEST_RUN_FOLDER' variable.")
        return

    os.makedirs(DESTINATION_FOLDER, exist_ok=True)
    print(f"Created destination folder: '{DESTINATION_FOLDER}'")

    train_files = {
        "results.csv": "results.csv",
        "results.png": "training_curves.png", # Contains all training curves
        os.path.join('weights', 'best.pt'): "best.pt"
    }

    # Files from the TEST folder
    test_files = {
        "confusion_matrix.png": "confusion_matrix_TEST.png",
        "val_batch0_pred.jpg": "sample_prediction_1.jpg",
        "val_batch1_pred.jpg": "sample_prediction_2.jpg",
        "val_batch2_pred.jpg": "sample_prediction_3.jpg"
    }

    # --- Copying Process ---
    copied_count = 0
    
    def copy_file(source_path, dest_name):
        nonlocal copied_count
        dest_path = os.path.join(DESTINATION_FOLDER, dest_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            print(f"✅ Copied: {dest_name}")
            copied_count += 1
        else:
            print(f"⚠️ Not found: {source_path}")

    print("\n--- Copying files from TRAINING folder ---")
    for src, dest in train_files.items():
        copy_file(os.path.join(TRAIN_RUN_FOLDER, src), dest)

    print("\n--- Copying files from TEST EVALUATION folder ---")
    for src, dest in test_files.items():
        copy_file(os.path.join(TEST_RUN_FOLDER, src), dest)

    print("\n--- Extraction Complete ---")
    print(f"Successfully copied {copied_count} files to the '{DESTINATION_FOLDER}' folder.")

if __name__ == "__main__":
    extract_submission_files()