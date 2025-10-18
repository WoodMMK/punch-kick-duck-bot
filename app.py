import cv2
import numpy as np
import pyautogui
import mss
import time

# --- โหลดภาพศัตรูแต่ละชนิด (1 ภาพต่อชนิด) แบบสีเต็ม ---
enemy_punch_right = cv2.imread("enemy/enemy_punch1.png", cv2.IMREAD_COLOR)
enemy_kick_right = cv2.imread("enemy/enemy_kick1.png", cv2.IMREAD_COLOR)
enemy_duck_right = cv2.imread("enemy/enemy_duck1.png", cv2.IMREAD_COLOR)

# ฝั่งซ้ายใช้ภาพ flip แนวนอน
enemy_punch_left = cv2.flip(enemy_punch_right, 1)
enemy_kick_left = cv2.flip(enemy_kick_right, 1)
enemy_duck_left = cv2.flip(enemy_duck_right, 1)

THRESHOLD = 0.6

sct = mss.mss()

# --- กำหนดพื้นที่ตรวจจับตามที่บอก ---
left_zone = {"top": 200, "left": 1150, "width": 150, "height": 200}   
right_zone = {"top": 200, "left": 1400, "width": 150, "height": 200}

# --- พิกัดคลิกสำหรับแต่ละปุ่ม ---
click_positions = {
    "A": (910, 3190),
    "S": (910, 4365),
    "D": (910, 5600),
    "J": (1820, 360),
    "K": (1820, 465),
    "L": (1820, 570),
}

print("starting...")

while True:
    #hold running button
    #

    # --- ตรวจฝั่งซ้าย ---
    img_left = np.array(sct.grab(left_zone))
    frame_left = cv2.cvtColor(img_left, cv2.COLOR_BGRA2BGR)

    for template, key in [(enemy_punch_left, "A"), (enemy_kick_left, "S"), (enemy_duck_left, "D")]:
        res = cv2.matchTemplate(frame_left, template, cv2.TM_CCOEFF_NORMED)
        if cv2.minMaxLoc(res)[1] > THRESHOLD:
            print(f"Clicking {key} for left enemy")
            pyautogui.click(*click_positions[key])
            time.sleep(0.05)

    # --- ตรวจฝั่งขวา ---
    img_right = np.array(sct.grab(right_zone))
    frame_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2BGR)

    for template, key in [(enemy_punch_right, "J"), (enemy_kick_right, "K"), (enemy_duck_right, "L")]:
        res = cv2.matchTemplate(frame_right, template, cv2.TM_CCOEFF_NORMED)
        if cv2.minMaxLoc(res)[1] > THRESHOLD:
            print(f"Clicking {key} for right enemy")
            pyautogui.click(*click_positions[key])
            time.sleep(0.05)

    time.sleep(0.05)
