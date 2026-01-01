#!/usr/bin/env python
"""
Run Sat2Graph inference on custom satellite images.
Processes images from sat_images/ folder and saves results to custom_outputs/
"""

import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from time import time

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from model import Sat2GraphModel
from decoder import DecodeAndVis

def load_image(image_path):
    """Load and preprocess a satellite image."""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32)
    
    # Normalize to [-0.45, 0.45] range (matching the training data preprocessing)
    max_v = np.amax(img) + 0.0001
    img = (img / max_v - 0.5) * 0.9
    
    # Add batch dimension
    img = img.reshape((1, 2048, 2048, 3))
    return img

def run_inference(model, sat_img, output_prefix):
    """Run inference on a single image."""
    print(f"Processing {output_prefix}...")
    
    image_size = 352
    max_degree = 6
    
    # Create output arrays
    output = np.zeros((2048+64, 2048+64, 2+4*6 + 2))
    mask = np.ones((2048+64, 2048+64, 2+4*6 + 2)) * 0.001
    
    # Define weights for overlapping regions
    weights = np.ones((image_size, image_size, 2+4*6 + 2)) * 0.001 
    weights[32:image_size-32, 32:image_size-32, :] = 0.5 
    weights[56:image_size-56, 56:image_size-56, :] = 1.0 
    weights[88:image_size-88, 88:image_size-88, :] = 1.5 
    
    # Pad input
    input_sat = np.pad(sat_img, ((0,0),(32,32),(32,32),(0,0)), 'constant')
    
    # Placeholder ground truth (not used in inference, but required by model)
    # gt_prob and gt_vector are padded to match input_sat
    gt_prob = np.zeros((1, 2048+64, 2048+64, 2 * (max_degree + 1)))  # 14 channels
    gt_vector = np.zeros((1, 2048+64, 2048+64, 2 * max_degree))  # 12 channels
    # gt_seg remains at window size (not padded)
    gt_seg = np.zeros((1, image_size, image_size, 1))
    
    # Sliding window inference
    t0 = time()
    for x in range(0, 352*6-352, 176//2):
        progress = int(x//88)
        sys.stdout.write(f"\r  Progress: {'>' * progress}{'.' * (20-progress)}")
        sys.stdout.flush()
        
        for y in range(0, 352*6-352, 176//2):
            alloutputs = model.Evaluate(
                input_sat[:, x:x+image_size, y:y+image_size, :], 
                gt_prob[:, x:x+image_size, y:y+image_size, :], 
                gt_vector[:, x:x+image_size, y:y+image_size, :], 
                gt_seg
            )
            _output = alloutputs[1]
            
            mask[x:x+image_size, y:y+image_size, :] += weights
            output[x:x+image_size, y:y+image_size, :] += np.multiply(_output[0,:,:,:], weights)
    
    print(f"\r  Inference time: {time()-t0:.2f} seconds")
    
    # Normalize output
    output = np.divide(output, mask)
    output = output[32:2048+32, 32:2048+32, :]
    input_sat_cropped = input_sat[:, 32:2048+32, 32:2048+32, :]
    
    # Save keypoints visualization
    output_keypoints_img = (output[:, :, 0] * 255.0).reshape((2048, 2048)).astype(np.uint8)
    Image.fromarray(output_keypoints_img).save(f"{output_prefix}_keypoints.png")
    
    # Save input image
    input_sat_img = ((input_sat_cropped[0,:,:,:] + 0.5) * 255.0).reshape((2048, 2048, 3)).astype(np.uint8)
    Image.fromarray(input_sat_img).save(f"{output_prefix}_input.png")
    
    # Decode to graph
    print("  Decoding to graph...")
    graph = DecodeAndVis(output, output_prefix, thr=0.05, edge_thr=0.05, snap=True, imagesize=2048)
    
    print(f"  Extracted {len(graph)} nodes\n")
    return graph

def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'sat_images')
    output_dir = os.path.join(script_dir, 'custom_outputs')
    model_path = os.path.join(script_dir, 'data/20citiesModel/model')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} images to process\n")
    
    # Initialize TensorFlow session and model
    print("Loading model...")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = Sat2GraphModel(
            sess, 
            image_size=352, 
            resnet_step=8, 
            batchsize=1, 
            channel=12, 
            mode='test'
        )
        
        # Load pretrained weights
        model.saver.restore(sess, model_path)
        print("Model loaded\n")
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            img_name = os.path.splitext(img_file)[0]
            output_prefix = os.path.join(output_dir, img_name)
            
            # Load image
            sat_img = load_image(img_path)
            
            # Run inference
            graph = run_inference(model, sat_img, output_prefix)
    
    print(f"\nAll done! Results saved to {output_dir}/")

if __name__ == '__main__':
    main()
