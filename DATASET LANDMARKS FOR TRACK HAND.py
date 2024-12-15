import cv2
import mediapipe as mp
import numpy as np
import csv

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Capture video feed from IP webcam
cap = cv2.VideoCapture('http://192.168.163.75:8080/video')

# Initialize a file to save landmarks as CSV
file_name = 'Landmark.csv'

# Prompt for the gesture label
gesture_label = input("Enter the gesture label (e.g., 'A', 'B'): ")

# Open the CSV file for writing
with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write header
    header = ['Gesture'] + [f'Landmark_{i+1}_X' for i in range(21)] + \
             [f'Landmark_{i+1}_Y' for i in range(21)] + \
             [f'Landmark_{i+1}_Z' for i in range(21)]
    writer.writerow(header)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform hand detection
        results = hands.process(image_rgb)
        
        # Extract landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Collect the landmark coordinates into a list
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                # Convert the landmarks to a flat array for saving
                landmarks = np.array(landmarks).flatten()
                print(landmarks)  # Print the feature vector for each frame

        # Display the processed frame
        cv2.imshow('Hand Tracking', frame)
        
        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF
        
        # Save the landmarks if 's' is pressed
        if key == ord('s') and len(landmarks) > 0:
            # Write the gesture label followed by the landmarks
            row_data = [gesture_label] + landmarks.tolist()  # Combine label with landmarks
            writer.writerow(row_data)  # Save as a row in the CSV file
            print("Landmarks saved to file.")

        # Exit on pressing 'q'
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
