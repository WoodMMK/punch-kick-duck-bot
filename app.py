import cv2
import numpy as np
import pyautogui
import mss
import time
import win32gui
import sys # <-- Import sys to exit on failure
import keyboard
from ultralytics import YOLO

# --- Load Model ---
try:
    model = YOLO("detect_enemy_1.pt")
    model_threshold = 0.4
    print("detection model loaded")
except Exception as e:
    print(f"[FATAL ERROR] Could not load model 'detect_enemy.pt': {e}")
    sys.exit()

def findWindow(window_name="BlueStacks App Player"):
    """Finds the window"""
    print(f"Attempting to find window: '{window_name}'...")
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        print(f"\n[ERROR] Could not find window: '{window_name}'.")
        print("Is BlueStacks running? Is the name exactly correct?")
        return None, None

    rect = win32gui.GetWindowRect(hwnd)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    
    window_frame = {"top": y, "left": x, "width": w, "height": h}
    sct_instance = mss.mss()
    print(f"Window found at ({x}, {y}) with size ({w}x{h})")
    return sct_instance, window_frame


def Cannyimg(img):
    """Converts an image to Canny edges."""
    # Check if image has alpha channel (4 channels)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # Convert to 3-channel BGR
        
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_edges = cv2.Canny(img_blur, 100, 200) 
    return img_edges

def findPosition(template_edges, screenshot_edges, threshold):
    """Finds the center of a template in a screenshot."""
    h, w = template_edges.shape[:2]
    result = cv2.matchTemplate(screenshot_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return (center_x, center_y)
    
    return None 

# --- 4. Calibrate All Buttons (NEW Anchor Version) ---
def findButton(sct, windowframe):
    """
    Finds the 2 middle buttons (hkl, hkr) and calculates the other 4.
    """
    print("\n--- Starting Button Calibration (Anchor Mode) ---")
    threshold = 0.4 # Start with 0.4 for edge matching

    # --- *** YOU MUST UPDATE THESE VALUES *** ---
    # Open a screenshot in MS Paint and measure the pixel distance
    # from the CENTER of the high-kick button to the CENTER of the others.
    
    # Offset from high-kick (middle) to punch (top)
    OFFSET_Y_PUNCH = -110 # <--- PUT YOUR NUMBER HERE
    # Offset from high-kick (middle) to low-kick (bottom)
    OFFSET_Y_KICK = 110  # <--- PUT YOUR NUMBER HERE
    # --- *** ---

    # Take one screenshot for calibration
    try:
        screenshot = np.array(sct.grab(windowframe))
        screenshot_edges = Cannyimg(screenshot)
    except Exception as e:
        print(f"[ERROR] Could not take screenshot: {e}")
        return None

    # Define search zones
    h, w = screenshot_edges.shape[:2]
    left_zone_rect = (0, h // 2, w // 2, h // 2) 
    right_zone_rect = (w // 2, h // 2, w // 2, h // 2) 

    left_search_edges = screenshot_edges[left_zone_rect[1]:left_zone_rect[1] + left_zone_rect[3],left_zone_rect[0]:left_zone_rect[0] + left_zone_rect[2]]
                                         
    right_search_edges = screenshot_edges[right_zone_rect[1]:right_zone_rect[1] + right_zone_rect[3], right_zone_rect[0]:right_zone_rect[0] + right_zone_rect[2]]

    # Define ONLY the anchor templates
    template_files = {
        "hkl": ("button/highKick_l.png", 'left'),
        "hkr": ("button/highKick_r.png", 'right')
    }
    
    button_positions_relative = {}
    all_found = True

    print("Searching for ANCHOR buttons (hkl, hkr)...")
    for key, (path, zone) in template_files.items():
        try:
            template_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if template_img is None:
                print(f"[ERROR] Failed to load template: {path}")
                all_found = False
                continue
                
            template_edges = Cannyimg(template_img)
            
            pos = None
            if zone == 'left':
                pos = findPosition(template_edges, left_search_edges, threshold)
                if pos:
                    pos = (pos[0] + left_zone_rect[0], pos[1] + left_zone_rect[1])
            elif zone == 'right':
                pos = findPosition(template_edges, right_search_edges, threshold)
                if pos:
                    pos = (pos[0] + right_zone_rect[0], pos[1] + right_zone_rect[1])
            
            if pos:
                button_positions_relative[key] = pos
                print(f"[SUCCESS] Found anchor '{key}' at window position {pos}")
            else:
                print(f"[FAIL] Could not find anchor '{key}'.")
                all_found = False

        except Exception as e:
            print(f"[ERROR] processing {path}: {e}")
            all_found = False

    if not all_found:
        print("[CRITICAL] Calibration failed. Bot cannot start.")
        return None

    # --- NEW: Calculate the other 4 buttons ---
    print("Calculating remaining button positions...")
    try:
        # Get the anchor positions
        hkl_pos = button_positions_relative["hkl"]
        hkr_pos = button_positions_relative["hkr"]
        
        # Calculate left buttons
        button_positions_relative["pl"] = (hkl_pos[0], hkl_pos[1] + OFFSET_Y_PUNCH)
        button_positions_relative["kl"] = (hkl_pos[0], hkl_pos[1] + OFFSET_Y_KICK)
        
        # Calculate right buttons
        button_positions_relative["pr"] = (hkr_pos[0], hkr_pos[1] + OFFSET_Y_PUNCH)
        button_positions_relative["kr"] = (hkr_pos[0], hkr_pos[1] + OFFSET_Y_KICK)
        
        print(f"  Calculated 'pl' at {button_positions_relative['pl']}")
        print(f"  Calculated 'kl' at {button_positions_relative['kl']}")
        print(f"  Calculated 'pr' at {button_positions_relative['pr']}")
        print(f"  Calculated 'kr' at {button_positions_relative['kr']}")

    except KeyError:
        print("[ERROR] Could not calculate positions because an anchor button was missing.")
        return None
    except Exception as e:
        print(f"[ERROR] in position calculation: {e}")
        return None

    # Convert relative (x,y) to absolute screen (x,y)
    window_x = windowframe["left"]
    window_y = windowframe["top"]
    
    button_positions_absolute = {}
    for key, (x, y) in button_positions_relative.items():
        button_positions_absolute[key] = (window_x + x, window_y + y)
    
    # --- NEW DEBUG PRINT ---
    print("\n--- Final Button Coordinates (Absolute) ---")
    for key, pos in button_positions_absolute.items():
        print(f"  {key}: {pos}")
    # --- END DEBUG PRINT ---
        
    print("--- Calibration SUCCESSFUL ---")
    return button_positions_absolute

# --- MAIN SCRIPT EXECUTION (OPTIMIZED WITH COOLDOWNS) ---
print("starting...")
sct, windowframe = findWindow()

if sct is None:
    sys.exit() # Exit if window not found

# Run calibration ONCE at the start
button_positions = findButton(sct, windowframe)

if button_positions is None:
    sys.exit() # Exit if calibration failed

# --- 1. SET UP YOUR MODEL & CLASSES ---
# (Already loaded model at the top)

# --- 2. DEFINE YOUR DYNAMIC ZONES & PLAYER ---
PLAYER_CLASS_NAME = "duck" 
DANGER_ZONE_BUFFER = 20  
MAX_DANGER_DISTANCE = 250 
RUN_KEY = 'shift' # <--- UPDATE THIS if it's a different key
default_player_x = windowframe["width"] // 2

# --- NEW: Independent Cooldowns ---
ATTACK_COOLDOWN = 0.3 # Cooldown in seconds. TUNE THIS.
last_attack_timestamp_left = 0
last_attack_timestamp_right = 0
# ---

# --- REMOVED: Create the debug window ---
cv2.namedWindow("Bot Debug View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bot Debug View", windowframe["width"] // 2, windowframe["height"] // 2)
cv2.moveWindow("Bot Debug View", windowframe["left"], windowframe["top"] + windowframe["height"] + 10)

print("\n--- Bot is RUNNING --- Press 'q' in this terminal to stop.")
print(f"Bot will look for player class: '{PLAYER_CLASS_NAME}'")
print(f"Danger zone set to: {DANGER_ZONE_BUFFER}px to {MAX_DANGER_DISTANCE}px from player.")

# --- 3. OPTIMIZED MAIN LOOP ---
while True:
    if keyboard.is_pressed('q'):
        pyautogui.keyUp(RUN_KEY) 
        print("Quitting...")
        break

    current_time = time.time() # Get time at the start of the loop
        
    # --- 4. GRAB ONCE, PREDICT ONCE ---
    img = np.array(sct.grab(windowframe))
    frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # --- REMOVED: Create a copy of the frame to draw on ---
    debug_frame = frame_bgr.copy()
    h, w, _ = debug_frame.shape

    # Run YOLOv8 detection ONCE on the full frame
    results = model(frame_bgr, conf=model_threshold, verbose=False, stream=True)

    enemy_detected_left = False # Reset flag
    enemy_detected_right = False # Reset flag
    current_player_x = default_player_x # Reset to default each frame
    
    detections = []
    try:
        for r in results:
            detections.extend(r.boxes)
    except Exception as e:
        print(f"Error processing results: {e}")
        continue # Skip this frame

    # --- 5. PASS 1: Find the Player (This is UNCOMMENTED and fixed) ---
    for box in detections:
        try:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == PLAYER_CLASS_NAME:
                x1, y1, x2, y2 = box.xyxy[0]
                current_player_x = (x1 + x2) / 2 # Get player's center X
                
                # --- REMOVED: Draw GREEN box on the duck ---
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(debug_frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(debug_frame, "PLAYER", (pt1[0], pt1[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break # Found the player, stop this loop
        except:
            continue # Ignore bad detections

    # --- 6. PASS 2: Find Enemies in the *New Smaller Zones* ---
    
    # Define the dynamic zones based on the player's *current* position
    left_danger_zone_start = int(current_player_x - MAX_DANGER_DISTANCE)
    left_danger_zone_end = int(current_player_x - DANGER_ZONE_BUFFER)
    right_danger_zone_start = int(current_player_x + DANGER_ZONE_BUFFER)
    right_danger_zone_end = int(current_player_x + MAX_DANGER_DISTANCE)
    
    # --- REMOVED: Draw danger zones ---
    cv2.rectangle(debug_frame, (left_danger_zone_start, 0), (left_danger_zone_end, h), (0, 255, 255), 2)
    cv2.rectangle(debug_frame, (right_danger_zone_start, 0), (right_danger_zone_end, h), (0, 255, 255), 2)

    # --- 7. NEW: Check Left Side (if not on cooldown) ---
    if current_time - last_attack_timestamp_left > ATTACK_COOLDOWN:
        for box in detections:
            try:
                cls_id = int(box.cls[0]); label = model.names[cls_id]; conf = float(box.conf[0])
            except: continue
            if label == PLAYER_CLASS_NAME or conf < model_threshold: continue
            
            x1, y1, x2, y2 = box.xyxy[0]; enemy_center_x = (x1 + x2) / 2

            # Check if enemy is in the LEFT zone
            if left_danger_zone_start < enemy_center_x < left_danger_zone_end:
                enemy_detected_left = True
                # --- REMOVED: Draw RED box on the enemy ---
                cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                if label == "yellowbottle" or label== "left bunny":
                    pyautogui.click(button_positions["pl"])
                    print(f"Left Zone: Detected {label} -> click pl")
                elif label == "cart" or label =="pig" or label== "redbottle":
                    pyautogui.click(button_positions["hkl"])
                    print(f"Left Zone: Detected {label} -> click hkl")
                elif label == "furret" or label=="greenbottle":
                    pyautogui.click(button_positions["kl"])
                    print(f"Left Zone: Detected {label} -> click kl")
                
                last_attack_timestamp_left = time.time() # START LEFT COOLDOWN
                break # Stop searching for *more* left enemies this frame
    
    # --- NEW: Check Right Side (if not on cooldown) ---
    if current_time - last_attack_timestamp_right > ATTACK_COOLDOWN:
        for box in detections:
            try:
                cls_id = int(box.cls[0]); label = model.names[cls_id]; conf = float(box.conf[0])
            except: continue
            if label == PLAYER_CLASS_NAME or conf < model_threshold: continue
            
            x1, y1, x2, y2 = box.xyxy[0]; enemy_center_x = (x1 + x2) / 2

            # Check if enemy is in the RIGHT zone
            if right_danger_zone_start < enemy_center_x < right_danger_zone_end:
                enemy_detected_right = True
                # --- REMOVED: Draw RED box on the enemy ---
                cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                if label == "yellowbottle" or label== "left bunny":
                    pyautogui.click(button_positions["pr"])
                    print(f"Right Zone: Detected {label} -> click pr")
                elif label == "cart" or label =="pig" or label== "redbottle":
                    pyautogui.click(button_positions["hkr"])
                    print(f"Right Zone: Detected {label} -> click hkr")
                elif label == "furret" or label=="greenbottle":
                    pyautogui.click(button_positions["kr"])
                    print(f"Right Zone: Detected {label} -> click kr")
                
                last_attack_timestamp_right = time.time() # START RIGHT COOLDOWN
                break # Stop searching for *more* right enemies this frame
    
    # --- 8. Handle "Hold Run" (Idle Action) ---
    if enemy_detected_left or enemy_detected_right:
        pyautogui.keyUp(RUN_KEY) # An enemy was detected, so stop running
    else:
        # Only hold run if BOTH sides are off cooldown
        if (current_time - last_attack_timestamp_left > ATTACK_COOLDOWN) and \
           (current_time - last_attack_timestamp_right > ATTACK_COOLDOWN):
            pyautogui.keyDown(RUN_KEY) # All clear, hold run

    # --- 9. Show debug frame and manage loop speed ---
    # --- REMOVED: Show the debug frame ---
    cv2.imshow("Bot Debug View", debug_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Allow closing window to quit
        break
    
    # --- RE-ADD: Small sleep, since cv2.waitKey(1) was removed ---
    time.sleep(0.01)

# --- Clean up ---
# --- REMOVED: Destroy the debug window ---
cv2.destroyAllWindows()
pyautogui.keyUp(RUN_KEY) # Ensure 'run' key is released on exit
print("Bot stopped.")