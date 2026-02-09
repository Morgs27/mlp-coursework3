import os
import glob
from gaze_tracking import process_image, INPUT_DIR

def main():
    image_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    print(f"Found {len(image_files)} images. processing all...")
    
    for img_path in image_files:
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {os.path.basename(img_path)}...")
            process_image(img_path, save_output=True)
            
    print("Done processing all images.")

if __name__ == "__main__":
    main()
