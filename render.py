#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" pip install ./diff-gaussian-rasterization
# TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" pip install ./simple-knn/

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
# from rembg import remove
import numpy as np
from PIL import Image
import pdb
import imageio

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    frames = []
    gts = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_torch = render(view[1].cuda(), gaussians, pipeline, background)["render"]
        gt = view[0][0:3, :, :]
        gt_numpy = gt.permute(1, 2, 0).cpu().numpy()
        
        rendering = rendering_torch.permute(1, 2, 0).cpu().numpy()
        frames.append(rendering)
        gts.append(gt_numpy)

        image_name = view[1].image_path.split('/')[-1].split('.')[0]


        # pdb.set_trace()
        # rendering.save(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering_torch, os.path.join(render_path, image_name + f"_{idx:03d}" + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, image_name + ".png"))
    
    imageio.mimsave(render_path+'video.mp4', [frame for frame in frames], fps=25)
    imageio.mimsave(render_path+'-gt.mp4', [frame for frame in gts], fps=25)

    print(render_path+'video.mp4')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1,1,1] 
        bg_color = [0.125,0.216,0.157]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)