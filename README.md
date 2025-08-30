# OralVis: YOLOv8 Tooth Detection and Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dental-xray-analysis.streamlit.app/)

This repository contains the complete project for the OralVis AI Research Intern Task. The objective is to train a YOLO model to detect and classify 32 types of teeth in panoramic dental X-rays using the FDI numbering system.

Through a series of experiments, the final and best-performing model was determined to be a **YOLOv8m** trained with strong data augmentation, achieving a final **mAP@50 of 0.905** on the held-out test set.

## Project Structure

```
├── data
├── .gitignore
├── data.yaml
├── model.py
├── extract_results.py
├── requirements.txt
├── split_data.py
└── README.md
```

---
## Step 1: Environment Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/parshv1234/OralVIS_Project.git
    cd OralVIS_Project
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---
## Step 2: Data Preparation

1.  **Place the Dataset**
    Place the provided `data` folder (containing all raw images and labels) directly into the root of the project directory.

2.  **Clean the Label Files**
    The initial dataset contains a few corrupted label files. Run the cleaning script to automatically fix these issues before proceeding.

    ```bash
    python3 cleaning.py
    ```

3.  **Split the Dataset**
    Run the splitting script to automatically divide the data into training (80%), validation (10%), and test (10%) sets. 
    ```bash
    python3 split_data.py
    ```
    This will create a new `OralVis_Dataset` folder with the structured data, ready for training.

---
## Step 3: Model Training

The `model.py` script is configured to train the best-performing model (`YOLOv8m`) with the optimal hyperparameters and data augmentations discovered during experimentation.

* **Run the Training Command** 
    ```bash
    python3 model.py
    ```
* **Results**
    The training process will save all results, including model weights (`best.pt`), training curves, and validation batches, into a new folder inside `runs/detect/`.

---
## Step 4: Final Evaluation

After training is complete, the final step is to evaluate the best model on the held-out test set to get the official performance metrics.

1.  **Identify Your Best Model**
    The best model weights are saved as `best.pt` inside your latest training run folder (e.g., `runs/detect/oralvis_yolov8m_/weights/best.pt`).

2.  **Run the Evaluation Command**
    Execute the following command, replacing `<YOUR_RUN_FOLDER_NAME>` with the name of your specific run folder.
    ```bash
    yolo task=detect mode=val model=runs/detect/<YOUR_RUN_FOLDER_NAME>/weights/best.pt data=data.yaml split=test
    ```

This command will output the final metrics and save the test set confusion matrix and sample predictions to a new folder (e.g., `runs/detect/validation_result/`). These are the final results for your report.

---
## Step 5: Post-Processing

Run the post-processing script on a sample image to see the final refined output.
```bash
python3 postprocess.py
```
This will generate a `post_processed_result.jpg` file in a new folder `postprocessing`.

---

## Final Model Performance

The final `YOLOv8m` model achieved the following performance on the test set:

| Metric      | Final Test Set Score |
| :---------- | :------------------- |
| **mAP@50** | **0.905** |
| mAP@50-95   | 0.614                |
| Precision   | 0.830                |
| Recall      | 0.881                |

---