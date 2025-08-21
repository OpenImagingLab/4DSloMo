import cv2
import os
import argparse
import shutil
from collections import defaultdict
from tqdm import tqdm

def crop_images(source_folder, output_folder):
    """
    Crops images from the source folder and saves them to the output folder.
    """
    print(f"Starting to crop images from '{source_folder}'...")
    os.makedirs(output_folder, exist_ok=True)

    filenames = [f for f in sorted(os.listdir(source_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    for filename in tqdm(filenames, desc=f"Cropping {os.path.basename(source_folder)}"):
        file_path = os.path.join(source_folder, filename)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Warning: Unable to load image {filename}. Skipping.")
            continue

        h, w = img.shape[:2]

        # Step 1: Scale while maintaining aspect ratio, making the smallest side 1024
        if w < h:
            new_w = 1024
            new_h = int(h * (1024 / w))
        else:
            new_h = 1024
            new_w = int(w * (1024 / h))

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Crop a 1024x1024 area from the center
        current_h, current_w = img_resized.shape[:2]
        start_x = (current_w - 1024) // 2
        start_y = (current_h - 1024) // 2
        
        img_cropped = img_resized[start_y:start_y + 1024, start_x:start_x + 1024]

        target_path = os.path.join(output_folder, filename)
        cv2.imwrite(target_path, img_cropped)

    print(f"Finished cropping. Cropped images are in '{output_folder}'.")

def create_videos_by_prefix(image_folder, output_folder, fps=25, max_frames=None):
    """
    Creates videos from sequences of images, grouped by filename prefix.
    """
    print(f"Starting to create videos from images in '{image_folder}'...")
    
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if not image_files:
        print("Error: No images found in the specified folder to create video from.")
        return

    # Group images by prefix (e.g., 'frame_0001' from 'frame_0001_1.png')
    image_groups = defaultdict(list)
    for image_file in image_files:
        prefix = image_file.split('_')[0]
        image_groups[prefix].append(image_file)
    
    if not image_groups:
        print("Error: Could not group any images by prefix.")
        return

    # Iterate over each group and generate a video
    for prefix, image_file_list in tqdm(image_groups.items(), desc="Processing video groups"):
        image_file_list.sort()
        
        if max_frames is not None and len(image_file_list) > max_frames:
            print(f"Trimming video for prefix '{prefix}' to {max_frames} frames.")
            image_file_list = image_file_list[:max_frames]
        
        first_image_path = os.path.join(image_folder, image_file_list[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            print(f"Error: Unable to read the first image for prefix '{prefix}'. Skipping group.")
            continue
        height, width, _ = first_image.shape

        output_video_path = os.path.join(output_folder, f'{prefix}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for image_file in image_file_list:
            image_path = os.path.join(image_folder, image_file)
            frame = cv2.imread(image_path)
            if frame is not None:
                video.write(frame)
            else:
                print(f"Warning: Skipping missing or corrupt frame {image_path}")

        video.release()
        print(f"Successfully created video for prefix '{prefix}': {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Crop images and create video(s). Supports a special mode for render outputs.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the source folder. If it contains "renders" and "gt" subfolders, special mode is activated.')
    parser.add_argument('--output_folder', type=str, help='(Generic mode only) Path to the folder where the final video(s) will be saved.')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second for the output video(s).')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames for each video. Defaults to all frames.')
    args = parser.parse_args()

    renders_path = os.path.join(args.input_folder, 'renders')
    gt_path = os.path.join(args.input_folder, 'gt')

    # Special mode for processing render outputs
    if os.path.isdir(renders_path) and os.path.isdir(gt_path):
        print("Detected 'renders' and 'gt' subfolders. Activating special render output mode.")
        
        # 1. Process 'renders' folder: crop images, create video, then cleanup
        print("\n--- Processing 'renders' folder ---")
        video_output_folder = os.path.join(args.input_folder, 'video_crop')
        temp_renders_crop_folder = os.path.join(video_output_folder, "temp_cropped_renders")
        
        try:
            os.makedirs(video_output_folder, exist_ok=True)
            crop_images(renders_path, temp_renders_crop_folder)
            if os.listdir(temp_renders_crop_folder):
                 create_videos_by_prefix(temp_renders_crop_folder, video_output_folder, args.fps, args.max_frames)
            else:
                 print("No images found in 'renders' folder to process.")
        finally:
            if os.path.exists(temp_renders_crop_folder):
                print(f"Cleaning up temporary folder: {temp_renders_crop_folder}")
                shutil.rmtree(temp_renders_crop_folder)

        # 2. Process 'gt' folder: crop images and save them directly
        print("\n--- Processing 'gt' folder ---")
        gt_crop_output_folder = os.path.join(args.input_folder, 'gt_crop')
        crop_images(gt_path, gt_crop_output_folder)

    # Generic mode (original functionality)
    else:
        print("Running in generic mode (no 'renders'/'gt' subfolders found).")
        if not args.output_folder:
            parser.error("--output_folder is required in generic mode.")

        temp_crop_folder = os.path.join(args.output_folder, "temp_cropped_images")
        os.makedirs(args.output_folder, exist_ok=True)
        
        try:
            crop_images(args.input_folder, temp_crop_folder)
            if os.listdir(temp_crop_folder):
                create_videos_by_prefix(temp_crop_folder, args.output_folder, args.fps, args.max_frames)
            else:
                print("No images found in source folder to process.")
        finally:
            if os.path.exists(temp_crop_folder):
                print(f"Cleaning up temporary folder: {temp_crop_folder}")
                shutil.rmtree(temp_crop_folder)

if __name__ == '__main__':
    main() 