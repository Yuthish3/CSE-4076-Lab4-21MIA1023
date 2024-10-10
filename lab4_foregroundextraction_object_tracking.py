import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Helper function to load grayscale frames from a directory
def load_frames_from_directory(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('L')  # Convert image to grayscale
            frames.append(np.array(image, dtype=np.float32))  # Load as float for averaging
    return frames

# Background Subtraction using the Averaging Method
def background_subtraction_averaging(current_frame, background, alpha=0.1):
    # Update the background with a weighted average
    updated_background = alpha * current_frame + (1 - alpha) * background
    # Calculate the absolute difference
    foreground_mask = np.abs(current_frame.astype(np.float32) - updated_background) > 30
    return foreground_mask.astype(np.uint8), updated_background

# Create an averaged background from multiple frames
def compute_initial_background(frames):
    return np.mean(frames, axis=0).astype(np.float32)

# Calculate the centroid of segmented regions
def calculate_centroid(binary_mask):
    coords = np.argwhere(binary_mask == 1)  # Get the indices of the foreground pixels
    if len(coords) == 0:
        return None  # No foreground detected
    centroid = np.mean(coords, axis=0)  # Calculate the centroid
    return centroid

# Save frames to a directory
def save_frame(frame, filename, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, filename)

    # Convert frame to uint8 format
    frame_uint8 = (frame * 255).astype(np.uint8)  # Convert to uint8 before saving
    Image.fromarray(frame_uint8).save(output_path)

# Main Processing Function
def spatio_temporal_segmentation(directory, output_directory, background_frame_count=3):
    frames = load_frames_from_directory(directory)
    
    if len(frames) < background_frame_count:
        print("Not enough frames for background averaging.")
        return
    
    # Compute the initial background using averaging over the first few frames
    background = compute_initial_background(frames[:background_frame_count])
    
    # Initialize previous centroid
    previous_centroid = None

    # Process each frame starting from the background frame count
    for i in range(background_frame_count, len(frames)):
        current_frame = frames[i]

        # Perform foreground extraction using the averaging method
        foreground_mask, background = background_subtraction_averaging(current_frame, background)

        # Calculate the centroid of the current foreground
        current_centroid = calculate_centroid(foreground_mask)

        if current_centroid is not None:
            if previous_centroid is not None:
                # Calculate movement
                movement = current_centroid - previous_centroid
                print(f"Frame {i}: Object moved by {movement}")

            previous_centroid = current_centroid  # Update the previous centroid

        # Apply the foreground mask to the original frame
        foreground_frame = np.where(foreground_mask, current_frame, 0)

        # Save the foreground frame
        save_frame(foreground_frame, f'foreground_{i:04d}.png', output_directory)



# Example usage
directory_path = "/Users/yuthishkumar/Desktop/python project/lab4frames_drink"  # Provide the path to the directory containing grayscale frames
output_directory = "/Users/yuthishkumar/Desktop/python project/foreground_frames"  # Provide the path to save foreground frames

spatio_temporal_segmentation(directory_path, output_directory)
