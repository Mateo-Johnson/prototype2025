from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model_path = "card-vision-model-output/weights/best.pt"
model = YOLO(model_path)  # Pretrained YOLOv8n model

def analyze_camera():
    cap = cv2.VideoCapture(6)  # Set the correct camera index for your system

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Run YOLO model inference on the frame
        results = model(frame, stream=True, verbose=False)  # Suppress internal logs

        for result in results:
            # Get bounding boxes, confidence scores, and class labels
            boxes = result.boxes.xyxy  # Coordinates of bounding boxes (x1, y1, x2, y2)
            scores = result.boxes.conf  # Confidence scores for each box
            labels = result.boxes.cls  # Class labels for the detected objects

            # Only process detections with confidence > 0.7
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score > 0.85:  # Only log detections with high confidence
                    x1, y1, x2, y2 = map(int, box)

                    # Get the class name
                    class_name = model.names[int(label)]

                    # Log the detected card information to the console
                    print(f"Detected {class_name} at [{x1}, {y1}, {x2}, {y2}] with confidence {score:.2f}")

                    # Draw a bounding box around the card for visualization (optional)
                    color = (0, 255, 0)  # Green bounding box
                    thickness = 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    # Optionally, add the label and confidence on the frame
                    text = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the frame for visualization (optional)
        cv2.imshow("Card-Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    analyze_camera()
