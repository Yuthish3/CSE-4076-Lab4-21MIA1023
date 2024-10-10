import cv2
import os

# Path to the video file
video_path = '/Users/yuthishkumar/Downloads/drinkad_iva_lab4.mp4'

# Directory where frames will be saved
output_dir = 'lab4_drinkad'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # If no frame is returned, end the loop

    # Save the original frame as an image
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)  # No color conversion, saving original frame

    frame_count += 1

# Release the video capture object
cap.release()

print(f'Extracted {frame_count} frames and saved them to {output_dir}')

