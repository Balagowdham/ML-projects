import cv2

# Use the IP address provided by the app
cap = cv2.VideoCapture('http://192.168.29.221:8080/video')

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Mobile Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
