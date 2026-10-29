
import cv2
import json
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(
    "facial_expression_model.keras"
)

# Load the expression labels
with open("class_names.json", "r") as file:
    class_names = json.load(file)

# Open the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

# Detect faces
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Face detector file could not be loaded.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = face.reshape(1, 48, 48, 1)

            probabilities = model.predict(face, verbose=0)[0]
            index = int(np.argmax(probabilities))

            expression = class_names[index]
            confidence = probabilities[index] * 100

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"{expression}: {confidence:.1f}%",
                (x, max(y-10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Facial Expression Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
