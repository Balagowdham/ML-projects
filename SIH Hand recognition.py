import cv2
import numpy as np
import os

# Create a directory to save the images if it doesn't exist
output_dir = 'saved_images'
os.makedirs(output_dir, exist_ok=True)

# Open the IP webcam stream
cap = cv2.VideoCapture('http://192.168.163.75:8080/video')

# Check if the video stream is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

image_count = 0  # To name saved images

print("Press 'S' to save the frame or 'Q' to quit.")

# Define the lower and upper boundaries for darker skin color in HSV
lower_skin = np.array([0, 30, 30], dtype=np.uint8)   # Adjusted for darker skin tone
upper_skin = np.array([20, 150, 150], dtype=np.uint8)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame was not captured correctly, break the loop
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for skin color detection
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Use morphological operations to remove small noises and obstacles
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are detected, process the largest one
    if contours:
        # Get the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Filter contours by size to avoid detecting small obstacles
        if cv2.contourArea(max_contour) > 1000:  # Adjust this threshold as needed
            # Draw a bounding box around the hand
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Optionally, highlight the hand contour itself
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('IP Webcam - Hand Highlighted', frame)

    # Wait for a key press (1ms delay)
    key = cv2.waitKey(1) & 0xFF

    # If 'S' is pressed, save the current frame
    if key == ord('s') or key == ord('S'):
        image_path = os.path.join(output_dir, f'image_{image_count}.png')
        cv2.imwrite(image_path, frame)
        print(f'Saved: {image_path}')
        image_count += 1

    # If 'Q' is pressed, quit the loop
    if key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
