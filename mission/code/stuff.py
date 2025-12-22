import cv2
import numpy as np
import pyvirtualcam

# Grid configuration
grid_config = {
    'num_cells': 4,
    'colors': [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)],  # Start with green (BGR format)
    'labels': ['?', '?', '?', '?'],
}

def draw_hand_grid(frame, colors, labels):
    """Draw a 1x4 grid at bottom of screen - identical to original"""
    grid_width = 640
    grid_height = 160
    grid_x = (640 - grid_width) // 2
    grid_y = 480 - grid_height - 10  # 10 pixels from bottom
    
    num_cells = len(colors)
    
    for i in range(num_cells):
        cell_width = grid_width // num_cells
        cell_x1 = grid_x + i * cell_width
        cell_x2 = cell_x1 + cell_width
        cell_y1 = grid_y
        cell_y2 = grid_y + grid_height
        
        # Draw filled rectangle with cell's color
        cv2.rectangle(frame, (cell_x1, cell_y1), (cell_x2, cell_y2), colors[i], -1)
        
        # Draw white border (same as original)
        cv2.rectangle(frame, (cell_x1, cell_y1), (cell_x2, cell_y2), (255, 255, 255), 2)
        
        # Draw label text
        text = labels[i]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = cell_x1 + (cell_width - text_size[0]) // 2
        text_y = cell_y1 + (grid_height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def toggle_cell(cell_index):
    """Toggle cell between green and red"""
    current_color = grid_config['colors'][cell_index]
    if current_color == (0, 255, 0):  # If green
        grid_config['colors'][cell_index] = (0, 0, 255)  # Set to red
        print(f"Cell {cell_index} -> RED")
    else:  # If red (or any other color)
        grid_config['colors'][cell_index] = (0, 255, 0)  # Set to green
        print(f"Cell {cell_index} -> GREEN")

def run_virtual_camera():
    # Open camera feed
    cap = cv2.VideoCapture(6)
    
    if not cap.isOpened():
        print("Error: Could not open camera 6")
        return
    
    with pyvirtualcam.Camera(width=640, height=480, fps=30, device="/dev/video11") as cam:
        print("Virtual Camera Grid Controller Started")
        print("Virtual camera device: /dev/video11")
        print("Camera feed: /dev/video6")
        print("\nControls:")
        print("  1 - Toggle cell 0 (green/red)")
        print("  2 - Toggle cell 1 (green/red)")
        print("  3 - Toggle cell 2 (green/red)")
        print("  4 - Toggle cell 3 (green/red)")
        print("  T - Edit text label")
        print("  Q - Quit")
        
        while cap.isOpened():
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Draw the grid (no other UI elements)
            draw_hand_grid(frame, grid_config['colors'], grid_config['labels'])
            
            # Send to virtual camera
            cam.send(frame)
            cam.sleep_until_next_frame()
            
            # Show frame and handle keyboard input
            cv2.imshow("Grid Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Process keyboard input
            if key == ord('q') or key == ord('Q'):
                print("Shutting down...")
                break
            
            elif key == ord('1'):
                toggle_cell(0)
            
            elif key == ord('2'):
                toggle_cell(1)
            
            elif key == ord('3'):
                toggle_cell(2)
            
            elif key == ord('4'):
                toggle_cell(3)
            
            elif key == ord('t') or key == ord('T'):  # Edit text
                cv2.destroyWindow("Grid Controller")
                cell_num = input("Which cell (0-3)? ")
                try:
                    cell_idx = int(cell_num)
                    if 0 <= cell_idx < 4:
                        new_label = input(f"Enter new label for cell {cell_idx}: ")
                        grid_config['labels'][cell_idx] = new_label
                        print(f"Cell {cell_idx} label updated to: {new_label}")
                    else:
                        print("Invalid cell number")
                except ValueError:
                    print("Invalid input")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_virtual_camera()