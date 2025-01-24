import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = preprocess_input(np.expand_dims(resized_frame, axis=0))

    # Predict
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]

    # Display predictions
    label = f"{decoded[0][1]}: {decoded[0][2]*100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Image Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
