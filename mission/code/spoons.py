import cv2
import numpy as np
from ultralytics import YOLO
import pyvirtualcam
from collections import Counter
import time
import os


from lerobot.utils.robot_utils import precise_sleep
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

episode_idx = 0

placeResume = False

robot_config = SO101FollowerConfig(port="/dev/ttyACM1", id="my_awesome_follower_arm")

robot = SO101Follower(robot_config)
robot.connect()


# Load the model
model_path = "card-vision-model-output/weights/best.pt"
model = YOLO(model_path)

# Game state for Spoons
game_state = {
    'player_cards': ['?', '?', '?', '?'],  # Cards currently in hand (start with placeholders)
    'current_card': None,  # Card being detected
    'spot_to_replace': -1,  # Position to replace (-1 = no replacement)
    'max_hand_size': 4,
    'check_signal': False,  # Flag for when to check/process card
    'next_slot_to_fill': 0,  # Track which slot to fill next during initial setup
}

# Card value rankings for Spoons strategy
card_values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

# ============================================================================
# ROBOT ACTION FUNCTIONS
# ============================================================================

def draw_and_bring_to_camera():
    """
    Robot action to:
    1. Draw a card from the deck
    2. Move the card to camera view for detection
    3. Hold steady for detection
    """
    print("[ROBOT ACTION] Drawing card and bringing to camera...")
    replayEpisode("scanCard5", 0)

def store_card_temp():
    """
    Robot action to:
    1. Store the currently held card in a temporary holding position
    2. Used when replacing a card from hand
    """
    print("[ROBOT ACTION] Storing card in temporary position...")
    replayEpisode("storeCardTemp11", 0)
    pass

def store_card_in_slot(position):
    """
    TODO: Implement robot action to:
    1. Take the card and store it in the specified slot position (0-3)
    2. This is for the initial hand filling phase
    
    Args:
        position: Index (0-3) of slot to store card in
    """
    print(f"[ROBOT ACTION] Storing card in slot {position}...")
    if (position == 0):
        replayEpisode("storeCardSlot0a3", 0)
    elif (position == 1):
         replayEpisode("storeCardSlot1a1", 0)
    elif (position == 2):
        replayEpisode("storeCardSlot2a1", 0)
    elif (position == 3):
        replayEpisode("storeCardSlot3a2", 0)
    # Add your robot control code here to store in specific slot
    pass

def replace_card_in_deck(position):
    """
    Robot action to:
    1. Take the card at the specified position from hand
    2. Place it in temporary storage, then pass it
    
    Args:
        position: Index (0-3) of card to replace in hand
    """
    print(f"[ROBOT ACTION] Replacing card at position {position}...")
    store_card_temp()
    print(f"[ROBOT ACTION] Grabbing card at position {position} to discard...")
    grab_card(position)
    pass_card()
    shift_temp(position)

def shift_temp(position):
    """
    Robot action to:
    1. Shift the temporarily held card to the discard position
    """
    print("[ROBOT ACTION] Shifting temporary card to discard position...")
    if (position == 0):
        replayEpisode("tempToSlot0a1", 0)
    elif (position == 1):
         replayEpisode("tempToSlot2a0", 0)
    elif (position == 2):
        replayEpisode("tempToSlot3a2", 0)
    elif (position == 3):
        replayEpisode("tempToSlot4a3", 0)

def grab_temp_card():
    """
    Robot action to:
    1. Grab the currently held card from camera position
    """
    print("[ROBOT ACTION] Grabbing card from camera...")
    replayEpisode("grabTempCard2", 0)

def grab_card(position):
    """
    Robot action to:
    1. Grab the card at the specified position from hand
    """
    print(f"[ROBOT ACTION] Grabbing card at position {position}...")
    if (position == 0):
        replayEpisode("grabSlot1a1", 0)
    elif (position == 1):
         replayEpisode("grabSlot2a1", 0)
    elif (position == 2):
        replayEpisode("grabSlot3a1", 0)
    elif (position == 3):
        replayEpisode("grabSlot4a1", 0)

def pass_card():
    """
    Robot action to:
    1. Pass the current card to the next player
    2. Do not add card to hand
    """
    print("[ROBOT ACTION] Passing card to next player...")
    playInference("discard1k", "discard")

def grab_spoon():
    """
    TODO: Implement robot action to:
    1. Reach for and grab a spoon from the center
    2. Signal victory!
    """
    print("[ROBOT ACTION] *** GRABBING SPOON! ***")
    # Add your robot control code here
    pass

# ============================================================================
# GAME LOGIC FUNCTIONS
# ============================================================================

def get_card_value(card_name):
    """Extract the value from card name (e.g., 'AS' -> 'A', '10H' -> '10')"""
    # Handle different card naming conventions
    if len(card_name) >= 2:
        if card_name[0:2] == '10':
            return '10'
        return card_name[0]
    return card_name

def evaluate_hand(hand):
    """
    Evaluate hand for Spoons strategy.
    Returns: (best_value, count) - the value with most cards and how many
    """
    if not hand:
        return None, 0
    
    # Filter out placeholder cards
    real_cards = [card for card in hand if card != '?']
    if not real_cards:
        return None, 0
    
    values = [get_card_value(card) for card in real_cards]
    value_count = Counter(values)
    most_common = value_count.most_common(1)[0]
    return most_common[0], most_common[1]

def spoons_decision(new_card, hand):
    """
    Decide whether to keep the new card and which position to replace.
    Strategy: Try to collect 4 of the same value
    
    Returns: (keep_new_card, position_to_replace)
    """
    # Check if we're still in initial fill phase
    if game_state['next_slot_to_fill'] < 4:
        slot = game_state['next_slot_to_fill']
        game_state['next_slot_to_fill'] += 1
        return True, slot
    
    new_value = get_card_value(new_card)
    
    # Count occurrences of each value in current hand
    hand_values = [get_card_value(card) for card in hand if card != '?']
    value_count = Counter(hand_values)
    
    # Count if we add the new card
    value_count_with_new = value_count.copy()
    value_count_with_new[new_value] += 1
    
    # Get best value in current hand
    current_best_value, current_best_count = evaluate_hand(hand)
    
    # Get best value if we keep new card
    best_value_with_new = max(value_count_with_new.items(), key=lambda x: x[1])
    
    # Decision logic
    if best_value_with_new[1] > current_best_count:
        # Keeping new card improves our hand
        # Find position of card that doesn't match our best set
        for i, card in enumerate(hand):
            if card == '?':
                continue
            card_value = get_card_value(card)
            if card_value != best_value_with_new[0]:
                return True, i
        
        # Shouldn't reach here, but if all match, replace first
        return True, 0
    
    elif best_value_with_new[1] == current_best_count and new_value == current_best_value:
        # New card matches our current best set
        # Replace a card not in the set
        for i, card in enumerate(hand):
            if card == '?':
                continue
            card_value = get_card_value(card)
            if card_value != current_best_value:
                return True, i
        
        # All cards already match
        return False, -1
    
    else:
        # New card doesn't improve hand, discard it
        return False, -1

def draw_hand_grid(frame, hand, spot_to_replace):
    """Draw a 1x4 grid showing current hand at bottom of screen"""
    grid_width = 640
    grid_height = 160
    grid_x = (640 - grid_width) // 2
    grid_y = 480 - grid_height - 10  # 10 pixels from bottom
    
    # Check for four of a kind
    best_value, count = evaluate_hand(hand)
    has_four_of_kind = (count == 4)
    
    for i in range(4):
        cell_width = grid_width // 4
        cell_x1 = grid_x + i * cell_width
        cell_x2 = cell_x1 + cell_width
        cell_y1 = grid_y
        cell_y2 = grid_y + grid_height
        
        # Color logic
        if has_four_of_kind:
            color = (255, 0, 0)  # Blue for four of a kind (BGR format)
        elif i == spot_to_replace:
            color = (0, 0, 255)  # Red for card being replaced
        elif i < len(hand):
            color = (0, 255, 0)  # Green for occupied slots
        else:
            color = (128, 128, 128)  # Gray for empty slots
        
        # Draw filled rectangle
        cv2.rectangle(frame, (cell_x1, cell_y1), (cell_x2, cell_y2), color, -1)
        
        # Draw border
        cv2.rectangle(frame, (cell_x1, cell_y1), (cell_x2, cell_y2), (255, 255, 255), 2)
        
        # Draw card text if slot is occupied
        if i < len(hand):
            card_text = hand[i]
            text_size = cv2.getTextSize(card_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = cell_x1 + (cell_width - text_size[0]) // 2
            text_y = cell_y1 + (grid_height + text_size[1]) // 2
            cv2.putText(frame, card_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def replayEpisode(repo, episode):
    dataset = LeRobotDataset("kaushikduddala/" + repo, episodes=[episode])
    actions = dataset.hf_dataset.select_columns("action")
    for idx in range(dataset.num_frames):
        t0 = time.perf_counter()

        action = {
            name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
        }
        robot.send_action(action)

        precise_sleep(1.0 / dataset.fps - (time.perf_counter() - t0))

def playInference(repo, dataset):
    os.system("rm -rf /home/jjjk/.cache/huggingface/lerobot/KaushikDuddala/eval_" + dataset)
    os.system('tmux -S ~/lerobot/tmux send "timeout 20 lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.cameras=\\"{wrist: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 11, width: 640, height: 480, fps: 30}}\\" --robot.id=my_awesome_follower_arm --display_data=false --dataset.repo_id=KaushikDuddala/eval_discard2 --dataset.single_task=\\"discard held card in box\\" --policy.path=KaushikDuddala/discard1k2" Enter')
    time.sleep(22)

def analyze_camera():
    cap = cv2.VideoCapture(10) # set to cam id
    placeResume = False
    waiting_for_card = False
    detection_start_time = None
    detection_delay = 1.0  # Wait 1 second after card is detected before processing

    
    with pyvirtualcam.Camera(width=640, height=480, fps=30, device="/dev/video12") as cam:
        with pyvirtualcam.Camera(width=640, height=480, fps=30, device="/dev/video13") as cam2:
            print("Spoons Robot Controller Started")
            print("Controls (in OpenCV window):")
            print("  D - Draw card and bring to camera (auto-processes after detection)")
            print("  SPACE - Manually process current card at camera")
            print("  R - Reset hand")
            print("  Q - Quit")
            
            while cap.isOpened():

                ret, frame = cap.read()
                
                # Create a copy for cam2 with white box
                frame_cam2 = frame.copy()
                # Draw 200x180 white box in bottom left corner
                cv2.rectangle(frame, (0, 300), (200, 480), (255, 255, 255), -1)
                cv2.rectangle(frame_cam2, (0, 300), (200, 480), (255, 255, 255), -1)

                
                cam2.send(frame_cam2)
                cam2.sleep_until_next_frame()
                if not ret:
                    break

                # Run YOLO detection
                results = model(frame, stream=True, verbose=False)
                
                detected_card = None
                
                # Process detection results
                for result in results:
                    boxes = result.boxes.xyxy
                    scores = result.boxes.conf
                    labels = result.boxes.cls
                    
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        if score > 0.7:
                            x1, y1, x2, y2 = map(int, box)
                            class_name = model.names[int(label)]
                            
                            # Store the detected card
                            if detected_card is None or score > detected_card[1]:
                                detected_card = (class_name, score)
                            
                            # Draw bounding box
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label_text = f"{class_name} {score:.2f}"
                            cv2.putText(frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw the hand grid
                draw_hand_grid(frame, game_state['player_cards'], game_state['spot_to_replace'])
                
                # Send to virtual camera
                cam.send(frame)
                cam.sleep_until_next_frame()
                
                # Show frame and handle keyboard input through cv2.waitKey
                cv2.imshow("Spoons Robot Controller", frame)
                key = cv2.waitKey(1) & 0xFF

                
                # Process keyboard input
                if key == ord('d') or key == ord('D'):  # D key
                    # ===== CALL: draw_and_bring_to_camera() =====
                    draw_and_bring_to_camera()
                    print("Card drawn and positioned at camera. Waiting for detection...")
                    waiting_for_card = True
                    detection_start_time = None
                
                # Auto-process logic: if waiting for card and card is detected
                if waiting_for_card and detected_card:
                    if detection_start_time is None:
                        detection_start_time = time.time()
                        print(f"Card detected: {detected_card[0]}. Processing in {detection_delay} seconds...")
                    
                    # Show countdown on frame
                    elapsed = time.time() - detection_start_time
                    remaining = detection_delay - elapsed
                    if remaining > 0:
                        countdown_text = f"Auto-processing in: {remaining:.1f}s"
                        cv2.putText(frame, countdown_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if time.time() - detection_start_time >= detection_delay:
                        # Auto-process the card
                        waiting_for_card = False
                        detection_start_time = None
                        card_name = detected_card[0]
                        game_state['current_card'] = card_name
                        
                        print(f"\n=== Auto-Processing Card ===")
                        print(f"Detected: {card_name} (confidence: {detected_card[1]:.2f})")
                        
                        # Make spoons decision
                        keep_card, position = spoons_decision(card_name, game_state['player_cards'])
                        
                        if keep_card:
                            if position >= 0:
                                # Replace card at position
                                old_card = game_state['player_cards'][position]
                                game_state['player_cards'][position] = card_name
                                
                                # Check if this is initial fill phase (card slot was empty)
                                if old_card == '?':
                                    print(f"Decision: KEEP {card_name} (filling position {position})")
                                    # ===== CALL: store_card_in_slot() for initial fill =====
                                    store_card_in_slot(position)
                                else:
                                    print(f"Decision: KEEP {card_name}, DISCARD {old_card} (position {position})")
                                    # ===== CALL: replace_card_in_deck() for replacement =====
                                    replace_card_in_deck(position)
                                    
                                
                                game_state['spot_to_replace'] = position
                        else:
                            print(f"Decision: PASS {card_name}")
                            game_state['spot_to_replace'] = -1
                            
                            # ===== CALL: pass_card() =====
                            pass_card()
                        
                        # Check for winning condition
                        best_value, count = evaluate_hand(game_state['player_cards'])
                        if count == 4:
                            print(f"\n*** FOUR OF A KIND: {best_value} - GRAB SPOON! ***")
                            
                            # ===== CALL: grab_spoon() =====
                            grab_spoon()
                        
                        print(f"Current Hand: {game_state['player_cards']}")
                
                elif key == ord(' '):  # Space key (manual processing)
                    placeResume = False
                    waiting_for_card = False
                    detection_start_time = None
                    if detected_card:
                        card_name = detected_card[0]
                        game_state['current_card'] = card_name
                        
                        print(f"\n=== Processing Card ===")
                        print(f"Detected: {card_name} (confidence: {detected_card[1]:.2f})")
                        
                        # Make spoons decision
                        keep_card, position = spoons_decision(card_name, game_state['player_cards'])
                        
                        if keep_card:
                            if position >= 0:
                                # Replace card at position
                                old_card = game_state['player_cards'][position]
                                game_state['player_cards'][position] = card_name
                                
                                # Check if this is initial fill phase (card slot was empty)
                                if old_card == '?':
                                    print(f"Decision: KEEP {card_name} (filling position {position})")
                                    # ===== CALL: store_card_in_slot() for initial fill =====
                                    store_card_in_slot(position)
                                else:
                                    print(f"Decision: KEEP {card_name}, DISCARD {old_card} (position {position})")
                                    # ===== CALL: replace_card_in_deck() for replacement =====
                                    replace_card_in_deck(position)
                                    
                                
                                game_state['spot_to_replace'] = position
                        else:
                            print(f"Decision: PASS {card_name}")
                            game_state['spot_to_replace'] = -1
                            
                            # ===== CALL: pass_card() =====
                            pass_card()
                        
                        # Check for winning condition
                        best_value, count = evaluate_hand(game_state['player_cards'])
                        if count == 4:
                            print(f"\n*** FOUR OF A KIND: {best_value} - GRAB SPOON! ***")
                            
                            # ===== CALL: grab_spoon() =====
                            grab_spoon()
                        
                        print(f"Current Hand: {game_state['player_cards']}")
                    else:
                        print("No card detected!")
                
                elif key == ord('r') or key == ord('R'):  # R key
                    game_state['player_cards'] = ['?', '?', '?', '?']
                    game_state['spot_to_replace'] = -1
                    game_state['current_card'] = None
                    game_state['next_slot_to_fill'] = 0
                    waiting_for_card = False
                    detection_start_time = None
                    print("\n=== Hand Reset ===")
                
                elif key == ord('q') or key == ord('Q'):  # Q key
                    print("Shutting down...")
                    break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    placeResume = False
    analyze_camera()
