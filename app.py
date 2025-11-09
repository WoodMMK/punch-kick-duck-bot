import cv2
import numpy as np
import pyautogui
import mss
import time
import win32gui
import sys # <-- Import sys to exit on failure
import keyboard
from ultralytics import YOLO

def findWindow(window_name="BlueStacks App Player"):
    """Finds the window and returns its coordinates and an mss instance."""
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

def findButton(sct, windowframe):
    """
    Finds all 6 buttons using ROI and returns their absolute screen coordinates.
    """
    print("\n--- Starting Button Calibration (with ROI) ---")
    threshold = 0.4 # Start with 0.4 for edge matching

    # Take one screenshot for calibration
    try:
        screenshot = np.array(sct.grab(windowframe))
        screenshot_edges = Cannyimg(screenshot)
    except Exception as e:
        print(f"[ERROR] Could not take screenshot: {e}")
        return None

    # --- NEW: Define Search Zones ---
    h, w = screenshot_edges.shape[:2]
    
    # Define rectangles (x, y, w, h) for the search zones
    # (x, y) is the top-left corner of the slice
    
    # Bottom-left quadrant of the window
    left_zone_rect = (0, h // 2, w // 2, h // 2) 
    # Bottom-right quadrant of the window
    right_zone_rect = (w // 2, h // 2, w // 2, h // 2) 

    # Slice the screenshot_edges to create two smaller search areas
    left_search_edges = screenshot_edges[left_zone_rect[1]:left_zone_rect[1] + left_zone_rect[3],left_zone_rect[0]:left_zone_rect[0] + left_zone_rect[2]]
                                         
    right_search_edges = screenshot_edges[right_zone_rect[1]:right_zone_rect[1] + right_zone_rect[3],right_zone_rect[0]:right_zone_rect[0] + right_zone_rect[2]]

    # Define all templates and which zone they belong in
    template_files = {
        "hkl": ("button/highKick_l.png", 'left'),
        "kl": ("button/kick_l.png", 'left'),
        "pl": ("button/punch_l.png", 'left'),
        "hkr": ("button/highKick_r.png", 'right'),
        "kr": ("button/kick_r.png", 'right'),
        "pr": ("button/punch_r.png", 'right')
    }
    
    button_positions_relative = {} # Stores (x,y) relative to the window
    all_found = True

    print("Searching for buttons in their correct zones...")
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
                    # ADD BACK THE OFFSET from the ROI slice
                    pos = (pos[0] + left_zone_rect[0], pos[1] + left_zone_rect[1])
            
            elif zone == 'right':
                pos = findPosition(template_edges, right_search_edges, threshold)
                if pos:
                    # ADD BACK THE OFFSET from the ROI slice
                    pos = (pos[0] + right_zone_rect[0], pos[1] + right_zone_rect[1])
            
            if pos:
                button_positions_relative[key] = pos
                print(f"[SUCCESS] Found '{key}' at window position {pos}")
            else:
                print(f"[FAIL] Could not find '{key}' in its zone. Try adjusting threshold.")
                all_found = False

        except Exception as e:
            print(f"[ERROR] processing {path}: {e}")
            all_found = False

    if not all_found:
        print("[CRITICAL] Calibration failed. Bot cannot start.")
        return None

    # Convert relative (x,y) to absolute screen (x,y)
    window_x = windowframe["left"]
    window_y = windowframe["top"]
    
    button_positions_absolute = {}
    for key, (x, y) in button_positions_relative.items():
        button_positions_absolute[key] = (window_x + x, window_y + y)
        
    print("--- Calibration SUCCESSFUL ---")
    return button_positions_absolute


# --- MAIN SCRIPT EXECUTION ---
print("starting...")
sct, windowframe = findWindow()

if sct is None:
    sys.exit() # Exit if window not found

# Run calibration ONCE at the start
button_positions = findButton(sct, windowframe)

if button_positions is None:
    sys.exit() # Exit if calibration failed

# --- NEW: Load ENEMY templates (with Mask) ---
print("Loading enemy templates with masks...")
try:
    # --- PUNCH ENEMY (Rabbit) ---
    # Load as-is, with transparency
    template_punch_rgba = cv2.imread("enemy/enemy_punch1.png", cv2.IMREAD_UNCHANGED)
    # Get the transparency channel (the 4th channel) to use as the mask
    template_punch_mask = template_punch_rgba[:, :, 3]
    # Get the color channels
    template_punch_bgr = template_punch_rgba[:, :, :3]
    # Flip them for the left side
    template_punch_bgr_left = cv2.flip(template_punch_bgr, 1)
    template_punch_mask_left = cv2.flip(template_punch_mask, 1)

    # --- KICK ENEMY (Pig) ---
    template_kick_rgba = cv2.imread("enemy/enemy_kick1.png", cv2.IMREAD_UNCHANGED)
    template_kick_mask = template_kick_rgba[:, :, 3]
    template_kick_bgr = template_kick_rgba[:, :, :3]
    template_kick_bgr_left = cv2.flip(template_kick_bgr, 1)
    template_kick_mask_left = cv2.flip(template_kick_mask, 1)

    # --- DUCK ENEMY (Ferret) ---
    template_duck_rgba = cv2.imread("enemy/enemy_duck1.png", cv2.IMREAD_UNCHANGED)
    template_duck_mask = template_duck_rgba[:, :, 3]
    template_duck_bgr = template_duck_rgba[:, :, :3]
    template_duck_bgr_left = cv2.flip(template_duck_bgr, 1)
    template_duck_mask_left = cv2.flip(template_duck_mask, 1)

except Exception as e:
    print(f"[ERROR] Could not load ENEMY template images: {e}")
    print("Make sure your enemy images are PNGs with a transparent background.")
    sys.exit()


# Define your enemy detection zones (relative to the window)
# *** YOU MUST UPDATE THESE VALUES. These are just guesses ***
# These zones MUST be in global screen coordinates for mss.grab()
zone_top_y = windowframe["top"] + 250  # Guess: 250px from top of window
zone_height = 200                     # Guess: 200px tall
left_zone_x = windowframe["left"] + 200 # Guess: 200px from left of window
right_zone_x = windowframe["left"] + 550 # Guess: 550px from left of window
zone_width = 150                      # Guess: 150px wide

left_zone = {"top": zone_top_y, "left": left_zone_x, "width": zone_width, "height": zone_height}
right_zone = {"top": zone_top_y, "left": right_zone_x, "width": zone_width, "height": zone_height}

# We need a HIGHER threshold now because the mask makes the match much more accurate
THRESHOLD = 0.48

print("\n--- Bot is RUNNING --- Press 'q' in this terminal to stop.")

# --- MAIN LOOP ---
while True:
    if keyboard.is_pressed('q'):
        print("Quitting...")
        break

    # --- ตรวจฝั่งซ้าย ---
    img_left = np.array(sct.grab(left_zone))
    frame_left = cv2.cvtColor(img_left, cv2.COLOR_BGRA2BGR)

    # --- NEW: Left side checks (with mask) ---
    res_punch_left = cv2.matchTemplate(frame_left, template_punch_bgr_left, cv2.TM_CCOEFF_NORMED, mask=template_punch_mask_left)
    score_punch_left = cv2.minMaxLoc(res_punch_left)[1]
    
    res_kick_left = cv2.matchTemplate(frame_left, template_kick_bgr_left, cv2.TM_CCOEFF_NORMED, mask=template_kick_mask_left)
    score_kick_left = cv2.minMaxLoc(res_kick_left)[1]
    
    res_duck_left = cv2.matchTemplate(frame_left, template_duck_bgr_left, cv2.TM_CCOEFF_NORMED, mask=template_duck_mask_left)
    score_duck_left = cv2.minMaxLoc(res_duck_left)[1]

    if score_punch_left > THRESHOLD:
        pyautogui.click(button_positions["pl"])
        time.sleep(0.1) # Wait a bit after a click
    elif score_kick_left > THRESHOLD:
        pyautogui.click(button_positions["hkl"])
        time.sleep(0.1)
    elif score_duck_left > THRESHOLD:
        pyautogui.click(button_positions["kl"])
        time.sleep(0.1)

    # --- ตรวจฝั่งขวา ---
    img_right = np.array(sct.grab(right_zone))
    frame_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2BGR)

    # --- NEW: Right side checks (with mask) ---
    res_punch_right = cv2.matchTemplate(frame_right, template_punch_bgr, cv2.TM_CCOEFF_NORMED, mask=template_punch_mask)
    score_punch_right = cv2.minMaxLoc(res_punch_right)[1]

    res_kick_right = cv2.matchTemplate(frame_right, template_kick_bgr, cv2.TM_CCOEFF_NORMED, mask=template_kick_mask)
    score_kick_right = cv2.minMaxLoc(res_kick_right)[1]

    res_duck_right = cv2.matchTemplate(frame_right, template_duck_bgr, cv2.TM_CCOEFF_NORMED, mask=template_duck_mask)
    score_duck_right = cv2.minMaxLoc(res_duck_right)[1]


    if score_punch_right > THRESHOLD:
        pyautogui.click(button_positions["pr"])
        time.sleep(0.1)
    elif score_kick_right > THRESHOLD:
        pyautogui.click(button_positions["hkr"])
        time.sleep(0.1)
    elif score_duck_right > THRESHOLD:
        pyautogui.click(button_positions["kr"])
        time.sleep(0.1)

    time.sleep(0.01) # Loop speed