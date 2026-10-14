
import cv2
import os
import time
from datetime import datetime

# IMPORTANT: Apne phone ka actual IP address daalo
STREAM_URL = "http://192.0.0.4:8080/video"

os.makedirs("snapshots", exist_ok=True)

camera = cv2.VideoCapture(STREAM_URL)

if not camera.isOpened():
    print("Camera connect nahi hua!")
    print("Phone ka IP address aur Wi-Fi check karo.")
    raise SystemExit

print("Camera connected successfully!")
print("Movement detect karne ke liye camera ke saamne haath hilao.")
print("Band karne ke liye video window par Q dabao.")

previous_gray = None
last_snapshot_time = 0

while True:
    success, frame = camera.read()

    if not success:
        print("Video nahi mil raha. Phone ka server check karo.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    motion_detected = False

    if previous_gray is not None:
        difference = cv2.absdiff(previous_gray, gray)

        _, threshold = cv2.threshold(
            difference, 25, 255, cv2.THRESH_BINARY
        )

        threshold = cv2.dilate(threshold, None, iterations=2)

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 1500:
                continue

            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

        if motion_detected:
            message = "MOTION DETECTED!"
            color = (0, 0, 255)

            if time.time() - last_snapshot_time >= 3:
                filename = datetime.now().strftime(
                    "motion_%Y%m%d_%H%M%S.jpg"
                )
                filepath = os.path.join("snapshots", filename)

                if cv2.imwrite(filepath, frame):
                    print("Snapshot saved:", filepath)
                    last_snapshot_time = time.time()
        else:
            message = "No Motion"
            color = (0, 255, 0)

        cv2.putText(
            frame, message, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

    previous_gray = gray.copy()
    cv2.imshow("Live Motion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("Motion detector stopped.")
