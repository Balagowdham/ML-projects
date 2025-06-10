import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = load_model('SIH/hand_sign_recognition_model.h5')  
classes = ['A', 'V', 'C']  
cap = cv2.VideoCapture('http://192.168.146.221:8080/video')


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w, _ = frame.shape
                x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
                x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
                y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
                y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

                hand_img = frame[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (1920, 1080))  

                hand_img = hand_img / 255.0  
                hand_img = np.expand_dims(hand_img, axis=0) 
                prediction = model.predict(hand_img)
                sign_index = np.argmax(prediction)  
                predicted_sign = classes[sign_index] 
                cv2.putText(frame, f'Sign: {predicted_sign}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Mobile Camera Feed - Sign Language Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
