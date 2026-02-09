import os
import cv2
from gaze_tracking import process_image, INPUT_DIR

def main():
    test_files = ["sv_a_19-1.png", "sv_a_20-1.png"]
    
    print("Generating verification images...")
    for filename in test_files:
        img_path = os.path.join(INPUT_DIR, filename)
        if os.path.exists(img_path):
            print(f"Processing {filename}...")
            process_image(img_path, save_output=True)
            print(f"Saved annotated image to annotated-images/{filename}")
        else:
            print(f"File {filename} not found.")

if __name__ == "__main__":
    main()
