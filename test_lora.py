import torch
import torch.distributed as dist
from diffsynth import ModelManager, save_video, VideoData
from FixModel import FixPipeline
from PIL import Image
import cv2
import os
import argparse
from typing import List

def save_img(video: List[Image.Image], output_folder: str, cam_name: str):
    """Saves a list of PIL Images to a directory."""
    images_dir = os.path.join(output_folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    for i, frame in enumerate(video):
        file_path = os.path.join(images_dir, f"{cam_name}_{i:04d}.png")
        frame.save(file_path)
        print(file_path)

    print(f"Saved {len(video)} images for camera {cam_name} to {images_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate video from image sequence.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output video folder')
    parser.add_argument('--model_path', type=str, default='./checkpoints/4DSloMo_LoRA.ckpt', help='Path to the LoRA model checkpoint')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"Running distributed inference on {world_size} GPUs.")

    image_folder = args.input_folder
    output_folder = args.output_folder

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(
        ["checkpoints/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,
    )

    model_manager.load_models(
        [
            [
                "checkpoints/diffusion_pytorch_model-00001-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00002-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00003-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00004-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00005-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00006-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            "checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
            "checkpoints/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_lora(args.model_path, lora_alpha=1.0)
    pipe = FixPipeline.from_model_manager(model_manager, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    cam_list = ["19305323","19305319","19305336","19305328","19305326","19305340","19305309","19305329","19224108","19305334","19305337","19305314"]

    cameras_for_this_rank = cam_list[rank::world_size]
    print(f"Rank {rank} is assigned {len(cameras_for_this_rank)} cameras.")

    for cam_name in cameras_for_this_rank:
        print(f"------ Rank {rank} processing {cam_name} ------")
        image_path = f"{image_folder}/test/ours_None/gt_crop/{cam_name}_0000.png"
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found for camera {cam_name}: {image_path}")
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open image for camera {cam_name}: {image_path}") from e

        video_path = f"{image_folder}/test/ours_None/video_crop/{cam_name}.mp4"
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found for camera {cam_name}: {video_path}")

        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for camera {cam_name}: {video_path}")

        video = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            video.append(pil_frame)

        cap.release()
        if len(video) == 0:
            raise ValueError(f"No frames could be read from video for camera {cam_name}: {video_path}")

        video = pipe(
            prompt="A girl dancing",
            negative_prompt="Vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static image, cluttered background, three legs, crowded background, walking backwards",
            input_image=image,
            input_video=video,
            num_inference_steps=args.num_inference_steps,
            seed=0,num_frames=33, tiled=True,
            height=1024,  width=1024,cam_id=0
        )
        save_img(video, output_folder, cam_name)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()