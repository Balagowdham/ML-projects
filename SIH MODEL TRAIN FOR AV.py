import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained model for sign recognition
model = load_model('SIH/hand_sign_recognition_model.h5')  # Model trained with 1080p images

# Define the classes (corresponding to signs in your dataset)
classes = ['A', 'V', 'C']  # Added 'C' sign

# Use the IP address provided by the app for capturing video
cap = cv2.VideoCapture('http://192.168.146.221:8080/video')

# Set the resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize MediaPipe hands detection
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (MediaPipe uses RGB images)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        result = hands.process(image)

        # Draw hand landmarks and perform sign recognition if hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the hand area from the frame
                h, w, _ = frame.shape
                x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
                x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
                y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
                y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

                # Crop the hand region and resize to model input shape
                hand_img = frame[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (1920, 1080))  # Resize to match model input

                # Normalize and prepare the image for prediction
                hand_img = hand_img / 255.0  # Normalize
                hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

                # Predict the sign
                prediction = model.predict(hand_img)
                sign_index = np.argmax(prediction)  # Get the index of the highest confidence prediction
                predicted_sign = classes[sign_index]  # Get the corresponding sign from the classes

                # Display the predicted sign on the video feed
                cv2.putText(frame, f'Sign: {predicted_sign}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the mobile camera feed with hand tracking and sign recognition
        cv2.imshow('Mobile Camera Feed - Sign Language Recognition', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
