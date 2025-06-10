import cv2
import os


output_dir = 'saved_images'
os.makedirs(output_dir, exist_ok=True)


cap = cv2.VideoCapture('http://192.168.146.221:8080/video')


if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

image_count = 0 

print("Press 'S' to save the frame or 'Q' to quit.")

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow('IP Webcam', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        image_path = os.path.join(output_dir, f'image_{image_count}.png')
        cv2.imwrite(image_path, frame)
        print(f'Saved: {image_path}')
        image_count += 1
    if key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break
cap.release()
cv2.destroyAllWindows()
