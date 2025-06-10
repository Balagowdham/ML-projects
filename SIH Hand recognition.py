import cv2
import numpy as np
import os

output_dir = 'saved_images'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture('http://192.168.163.75:8080/video')

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
image_count = 0 

print("Press 'S' to save the frame or 'Q' to quit.")


lower_skin = np.array([0, 30, 30], dtype=np.uint8)  
upper_skin = np.array([20, 150, 150], dtype=np.uint8)

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:  
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
    cv2.imshow('IP Webcam - Hand Highlighted', frame)
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
