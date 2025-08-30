import os
import random
import shutil
from sklearn.model_selection import train_test_split

IMAGE_SOURCE_DIR = 'data/images'
LABEL_SOURCE_DIR = 'data/labels'
DEST_DIR = 'OralVis_Dataset'

TEST_SIZE = 0.20 
VAL_SIZE = 0.50  
RANDOM_STATE = 42

def split_dataset():
    print("Looking for files in 'data/images' and 'data/labels'...")
    
    if not os.path.exists(IMAGE_SOURCE_DIR) or not os.path.exists(LABEL_SOURCE_DIR):
        print(f"Error: Source directories not found.")
        print(f"Please ensure you have '{IMAGE_SOURCE_DIR}' and '{LABEL_SOURCE_DIR}' folders.")
        return

    image_stems = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_SOURCE_DIR) if f.lower().endswith(('.jpg'))}
    label_stems = {os.path.splitext(f)[0] for f in os.listdir(LABEL_SOURCE_DIR) if f.lower().endswith('.txt')}

    paired_stems = list(image_stems.intersection(label_stems))
    
    if not paired_stems:
        print("\nError: No matching image-label pairs were found between the two directories.")
        return
        
    print(f"\nFound {len(paired_stems)} valid image-label pairs. Proceeding with split.")

    random.seed(RANDOM_STATE)
    random.shuffle(paired_stems)
    train_stems, temp_stems = train_test_split(paired_stems, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_stems, test_stems = train_test_split(temp_stems, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    
    print(f"Total paired files: {len(paired_stems)}")
    print(f"Training set: {len(train_stems)} files")
    print(f"Validation set: {len(val_stems)} files")
    print(f"Test set: {len(test_stems)} files")

    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DEST_DIR, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, 'labels', subset), exist_ok=True)

    def copy_files(stems_list, subset):
        for stem in stems_list:
            img_name = None
            for ext in ['.jpg']:
                if os.path.exists(os.path.join(IMAGE_SOURCE_DIR, stem + ext)):
                    img_name = stem + ext
                    break
            
            if img_name:
                shutil.copy(os.path.join(IMAGE_SOURCE_DIR, img_name), os.path.join(DEST_DIR, 'images', subset, img_name))
                shutil.copy(os.path.join(LABEL_SOURCE_DIR, stem + '.txt'), os.path.join(DEST_DIR, 'labels', subset, stem + '.txt'))

    copy_files(train_stems, 'train')
    copy_files(val_stems, 'val')
    copy_files(test_stems, 'test')

    print("\nDataset splitting complete.")

if __name__ == "__main__":
    split_dataset()