import os
import pandas as pd

DATA_DIR = 'data/labels' 
REPORT_FILE = os.path.join(DATA_DIR, '_label_validation_report.csv') 

def clean_label_files():
    if not os.path.exists(REPORT_FILE):
        print(f"Error: Report file not found at '{REPORT_FILE}'. Skipping cleaning step.")
        return
    try:
        report_df = pd.read_csv(REPORT_FILE)
        print(f"Loaded report. Found {len(report_df)} issues to fix.")
    except Exception as e:
        print(f"Error reading report file: {e}")
        return

    fixed_count = 0
    for index, row in report_df.iterrows():
        image_stem = row['image_stem']
        label_filename = f"{image_stem}.txt"
        label_filepath = os.path.join(DATA_DIR, label_filename)

        if not os.path.exists(label_filepath):
            print(f"- Warning: Label file not found, skipping: {label_filename}")
            continue

        try:
            with open(label_filepath, 'r') as f:
                lines = f.readlines()

            original_line_count = len(lines)
            good_lines = [line for line in lines if len(line.strip().split()) == 5]
            cleaned_line_count = len(good_lines)

            if original_line_count != cleaned_line_count:
                print(f"  - Fixing {label_filename}: Removed {original_line_count - cleaned_line_count} bad line(s).")
                with open(label_filepath, 'w') as f:
                    f.writelines(good_lines)
                fixed_count += 1
            else:
                print(f"  - Checked {label_filename}: No bad lines found to remove.")

        except Exception as e:
            print(f"- Error processing {label_filename}: {e}")

    print(f"\nCleaning complete. Fixed {fixed_count} out of {len(report_df)} reported files.")


if __name__ == "__main__":
    clean_label_files()