import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path

def extract_gei_from_video(video_path, output_path, model=None, device=None):
    """Extract a Gait Energy Image from a single video.

    The YOLO ``model`` and ``device`` can be passed in so callers can load
    them once and reuse them across many videos (much faster than
    re-instantiating the model per video).
    """
    print(f"\nProcessing video: {video_path.name}...")

    # Load YOLOv8 Segmentation Model only if not provided by the caller.
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model is None:
        model = YOLO('yolov8n-seg.pt')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    silhouettes = []

    # 2. Process Video Frame by Frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO inference, looking ONLY for 'person' (class 0)
        results = model(frame, classes=[0], verbose=False, device=device)
        
        for result in results:
            if result.masks is not None:
                mask = result.masks.data[0].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                binary_mask = (mask * 255).astype(np.uint8)
                
                boxes = result.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    
                    cropped_silhouette = binary_mask[y1:y2, x1:x2]
                    resized_silhouette = cv2.resize(cropped_silhouette, (64, 64))
                    silhouettes.append(resized_silhouette)
                break 

    cap.release()

    # 3. Generate the Gait Energy Image (GEI)
    if len(silhouettes) == 0:
        print("No people detected in the video!")
        return

    print(f"Extracted {len(silhouettes)} silhouette frames. Generating GEI...")
    
    silhouettes_array = np.array(silhouettes)
    gei = np.mean(silhouettes_array, axis=0).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(gei).save(output_path)
    print(f"✅ Saved GEI to: {output_path.name}")

if __name__ == "__main__":
    RAW_DATA_DIR = Path("./Raw_Video_Data")
    GALLERY_DIR = Path("./dataset/TRACE_Gallery")
    
    # Ensure Gallery directory exists
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RAW_DATA_DIR.exists():
        print(f"Error: {RAW_DATA_DIR} folder not found! Please create it and add your videos.")
        exit()

    print("Starting automated Phase 0 extraction...")

    # Load YOLO model ONCE here, then reuse across every video. Loading the
    # YOLO weights inside the per-video function (the previous behaviour)
    # was a major bottleneck — a single model load takes seconds and was
    # being repeated for every input video.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = YOLO('yolov8n-seg.pt')

    # Iterate through every person's folder in Raw_Video_Data
    for person_folder in RAW_DATA_DIR.iterdir():
        if person_folder.is_dir():
            # Format the name (e.g., "Anurag Sharma Ravulapally" -> "Anurag_Sharma_Ravulapally")
            person_name = person_folder.name.replace(" ", "_") 
            
                # Find all mp4 files in this person's folder
            for video_file in person_folder.glob("*.mp4"):
                video_name = video_file.stem # Gets "left_right" without the .mp4
                
                # Build the final output path
                output_filename = f"{person_name}_{video_name}.png"
                output_path = GALLERY_DIR / output_filename
                
                # ---> THE UPGRADE: Check if it already exists <---
                if output_path.exists():
                    print(f"⏩ Skipping {output_filename} (Already exists)")
                    continue # Skips to the next video without processing
                
                # If it doesn't exist, run the heavy extraction
                extract_gei_from_video(video_file, output_path,
                                       model=yolo_model, device=device)
                
    print("\n🎉 Phase 1 Complete! All raw videos converted to GEIs in TRACE_Gallery.")