import cv2
import pyvirtualcam
import numpy as np

def forward_camera_with_whitebox():
    """
    Reads from camera ID 6 and forwards to two virtual cameras:
    - /dev/video10: Original frame with white box
    - /dev/video11: Original frame with white box
    Both cameras get the same white box overlay at bottom left.
    """
    cap = cv2.VideoCapture(6)
    
    if not cap.isOpened():
        print("Error: Could not open camera 6")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera forwarder started")
    print("Reading from: Camera ID 6")
    print("Forwarding to: /dev/video10 and /dev/video11")
    print("White box coordinates: (0, 300) to (200, 480)")
    print("Press Ctrl+C to quit")
    
    with pyvirtualcam.Camera(width=640, height=480, fps=30, device="/dev/video10") as cam2:
        with pyvirtualcam.Camera(width=640, height=480, fps=30, device="/dev/video11") as cam:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Create a copy for cam2
                frame_cam2 = frame.copy()
                
                # Draw white box on both frames (bottom left corner)
                cv2.rectangle(frame, (0, 300), (200, 480), (255, 255, 255), -1)
                cv2.rectangle(frame_cam2, (0, 300), (200, 480), (255, 255, 255), -1)
                
                # Convert frame to RGB (instead of BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_cam2_rgb = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2RGB)
                
                # Send to virtual cameras in RGB format (not BGR)
                cam2.send(frame_cam2_rgb)
                cam2.sleep_until_next_frame()
                
                cam.send(frame_rgb)
                cam.sleep_until_next_frame()

                # Show the frame to see what we are sending (for debugging)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera forwarder stopped")

if __name__ == '__main__':
    forward_camera_with_whitebox()
