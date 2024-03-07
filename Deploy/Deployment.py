import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("C:/Users/Lam/Downloads/fire_detection_model_ver_demo_special.h5")

# Define a function to classify frames
def classify_frame(frame):
    # Preprocess the frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0  # Normalize pixel values
    frame = tf.expand_dims(frame, axis=0)  # Add batch dimension

    # Perform inference
    prediction = model.predict(frame)

    # Determine the class (fire or non-fire)
    if prediction[0][0] > 0.5:
        return True
    else:
        return False

# Open a video capture
cap = cv2.VideoCapture(0)  # You can specify a different source (e.g., a video file)

while True:
    ret, frame = cap.read()  # Read a frame from the video source
    if not ret:
        break

    # Classify the frame
    is_fire = classify_frame(frame)

    # Draw a bounding box if fire is detected
    if is_fire:
        # Define the coordinates for the rectangle (adjust as needed)
        # top_left = (150, 150)
        # bottom_right = (250, 250)
        
        # # Draw the rectangle on the frame
        # cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

        cv2.putText(frame, "Fire Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Fire", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the result
    cv2.imshow('Fire Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
