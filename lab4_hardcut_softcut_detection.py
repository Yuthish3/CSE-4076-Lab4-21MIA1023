import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to compute manual histogram for one channel
def compute_histogram(image, num_bins=256):
    hist = np.zeros(num_bins)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    return hist

# Function to normalize histogram
def normalize_histogram(hist, num_pixels):
    return hist / num_pixels

# Function to compare histograms using intersection
def compare_histograms_intersection(hist1, hist2):
    intersection = np.sum(np.minimum(hist1, hist2))
    return intersection

# Path to the video file
video_path = '/Users/yuthishkumar/Downloads/drinkad_iva_lab4.mp4'

# Directory where frames will be saved
output_dir = 'testframes'
os.makedirs(output_dir, exist_ok=True)

# Directory where histograms will be saved
histogram_dir = 'testhist'
os.makedirs(histogram_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Variables to store histograms for comparison
prev_hist_h = None
prev_hist_s = None
prev_hist_v = None
frame_count = 0
similarity_scores = []

# Lists to store soft and hard cut frame numbers
soft_cut_frames = []
hard_cut_frame = None  # We will store only one hard cut

# Threshold for soft cut detection
soft_cut_threshold = 0.5

# Variable to store the minimum similarity score and its corresponding frame
min_similarity = float('inf')
min_similarity_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End the loop if no frame is returned

    # Save the original frame as an image
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split into H, S, V channels
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)

    # Compute histograms for each channel
    hist_h = compute_histogram(h_channel)
    hist_s = compute_histogram(s_channel)
    hist_v = compute_histogram(v_channel)

    # Normalize histograms
    num_pixels = h_channel.size
    hist_h = normalize_histogram(hist_h, num_pixels)
    hist_s = normalize_histogram(hist_s, num_pixels)
    hist_v = normalize_histogram(hist_v, num_pixels)

    # Plot and save histograms for H, S, and V channels
    plt.figure(figsize=(10, 5))

    # Plot Histogram for H Channel
    plt.subplot(1, 3, 1)
    plt.plot(hist_h, color='red')
    plt.title(f'H Channel Histogram (Frame {frame_count})')

    # Plot Histogram for S Channel
    plt.subplot(1, 3, 2)
    plt.plot(hist_s, color='green')
    plt.title(f'S Channel Histogram (Frame {frame_count})')

    # Plot Histogram for V Channel
    plt.subplot(1, 3, 3)
    plt.plot(hist_v, color='blue')
    plt.title(f'V Channel Histogram (Frame {frame_count})')

    # Save the histogram plot
    hist_filename = os.path.join(histogram_dir, f'histogram_{frame_count:04d}.png')
    plt.savefig(hist_filename)
    plt.close()

    # If this is not the first frame, compare histograms with the previous frame
    if prev_hist_h is not None:
        h_intersection = compare_histograms_intersection(prev_hist_h, hist_h)
        s_intersection = compare_histograms_intersection(prev_hist_s, hist_s)
        v_intersection = compare_histograms_intersection(prev_hist_v, hist_v)

        # Average intersection score
        avg_intersection = (h_intersection + s_intersection + v_intersection) / 3.0
        similarity_scores.append(avg_intersection)
        print(f'Frame {frame_count} similarity score: {avg_intersection}')

        # Track the minimum similarity score for the hard cut
        if avg_intersection < min_similarity:
            min_similarity = avg_intersection
            min_similarity_frame = frame_count

        # Detect soft cuts (exclude hard cut frame)
        if avg_intersection < soft_cut_threshold:
            soft_cut_frames.append(frame_count)

    # Save the current histograms as the previous histograms for the next comparison
    prev_hist_h = hist_h
    prev_hist_s = hist_s
    prev_hist_v = hist_v

    frame_count += 1

# Release the video capture object
cap.release()

# Store the frame with the minimum similarity as the hard cut frame
hard_cut_frame = min_similarity_frame

# Remove the hard cut frame from the soft cut list, if it exists
if hard_cut_frame in soft_cut_frames:
    soft_cut_frames.remove(hard_cut_frame)

# Display the lists of soft and hard cut frame numbers
print("Soft cut frames:", soft_cut_frames)
print("Hard cut frame:", hard_cut_frame)

# Display the total number of cuts detected
print(f"Total soft cuts detected: {len(soft_cut_frames)}")
print(f"Hard cut detected at frame: {hard_cut_frame} with similarity score: {min_similarity}")
