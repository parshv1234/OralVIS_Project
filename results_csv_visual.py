import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_directory = 'plots'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: '{output_directory}'")

previous_best_map50 = 0.837

try:
    results_df = pd.read_csv('runs/detect/oralvis_yolov8m_run1/results.csv')
    results_df.columns = results_df.columns.str.strip()
    completed_epochs = len(results_df)
except Exception as e:
    print(f"Error reading or processing the CSV file: {e}")
    results_df = None

if results_df is not None and not results_df.empty:
    best_epoch_index = results_df['metrics/mAP50-95(B)'].idxmax()
    best_epoch_stats = results_df.loc[best_epoch_index]

    best_epoch_number = int(best_epoch_stats['epoch'])
    best_precision = best_epoch_stats['metrics/precision(B)']
    best_recall = best_epoch_stats['metrics/recall(B)']
    best_map50 = best_epoch_stats['metrics/mAP50(B)']
    best_map50_95 = best_epoch_stats['metrics/mAP50-95(B)']

    print(f"--- Analysis After {completed_epochs} Epochs (YOLOv11 Medium) ---")
    print(f"\n--- Best Model Performance (So Far) ---")
    print(f"The best performance is still at Epoch: {best_epoch_number + 1}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"mAP@50: {best_map50:.4f}")
    print(f"mAP@50-95: {best_map50_95:.4f}")

    print("\n--- Comparison to Previous Check-in ---")
    print(f"Current Best mAP@50: {best_map50:.4f} (vs. Previous Best: {previous_best_map50:.4f})")
    if best_map50 > previous_best_map50:
        print("Status: Performance has improved.")
    else:
        print("Status: Performance has not improved further. The model has likely peaked.")


    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['epoch'], results_df['train/box_loss'], label='Box Loss (Training)')
    plt.plot(results_df['epoch'], results_df['train/cls_loss'], label='Class Loss (Training)')
    plt.plot(results_df['epoch'], results_df['val/box_loss'], label='Box Loss (Validation)', linestyle='--')
    plt.plot(results_df['epoch'], results_df['val/cls_loss'], label='Class Loss (Validation)', linestyle='--')
    plt.title(f'Training and Validation Losses (First {completed_epochs} Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_losses_epoch_yolov8m.png')
    print("\nSaved training losses plot to 'training_losses_epoch.png'")

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP@50')
    plt.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP@50-95')

    plt.axvline(x=best_epoch_number, color='r', linestyle='--', label=f'Best Epoch ({best_epoch_number + 1})')
    plt.title(f'Model Performance (mAP) vs. Epochs (First {completed_epochs} Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/map_performance_epoch_yolov8m.png')
    print("Saved mAP performance plot to 'map_performance_epoch.png'")
else:
    print("The results.csv file is empty or could not be read. No analysis to perform.")