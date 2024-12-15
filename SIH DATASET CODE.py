import cv2
import os

# Create a directory to save the images if it doesn't exist
output_dir = 'saved_images'
os.makedirs(output_dir, exist_ok=True)

# Open the IP webcam stream
cap = cv2.VideoCapture('http://192.168.146.221:8080/video')

# Check if the video stream is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

image_count = 0  # To name saved images

print("Press 'S' to save the frame or 'Q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame was not captured correctly, break the loop
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the resulting frame
    cv2.imshow('IP Webcam', frame)

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
