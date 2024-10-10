import os
import numpy as np
from PIL import Image

# Roberts Edge Detection
def roberts_edge_detection(image):
    # Roberts Kernels
    roberts_x = np.array([[2, 0],
                          [0, -2]])

    roberts_y = np.array([[0, 2],
                          [-2, 0]])

    # Apply Roberts kernels using manual convolution
    grad_x = convolve(image, roberts_x)
    grad_y = convolve(image, roberts_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize magnitude to range [0, 255]
    magnitude = (magnitude / np.max(magnitude)) * 255

    return magnitude.astype(np.uint8)

# Convolution function
def convolve(image, kernel):
    # Get dimensions of the image and kernel
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Compute padding size (assuming kernel is square)
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the image to handle borders
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialize output image
    output = np.zeros_like(image)

    # Perform convolution
    for i in range(image_h):
        for j in range(image_w):
            # Element-wise multiplication of kernel and the image region
            region = padded_image[i:i + kernel_h, j:j + kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

# Load frames from directory
def load_frames_from_directory(directory):
    frames = []
    filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            frames.append(np.array(image))
            filenames.append(filename)
    return frames, filenames

# Save frames to a new directory
def save_frames_to_directory(frames, filenames, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for frame, filename in zip(frames, filenames):
        output_path = os.path.join(output_directory, filename)
        Image.fromarray(frame).save(output_path)

# Apply Roberts edge detection on a directory of frames
def apply_roberts_on_directory(input_directory, output_directory):
    frames, filenames = load_frames_from_directory(input_directory)

    processed_frames = []
    for frame in frames:
        edges = roberts_edge_detection(frame)
        processed_frames.append(edges)
    
    save_frames_to_directory(processed_frames, filenames, output_directory)

# Example usage
input_directory = "/Users/yuthishkumar/Desktop/python project/test"  # Replace with the directory of input frames
output_directory = "/Users/yuthishkumar/Desktop/python project/robert_edges"  # Replace with the directory to save edge-detected frames

apply_roberts_on_directory(input_directory, output_directory)
