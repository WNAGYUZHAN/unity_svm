import cv2
import mediapipe as mp
import joblib
import os
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
import socket   # <<< 新增

# ===================== UDP =====================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# ---------- 環境（可省略） ----------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ===== 顯示與功能開關 =====
MIRROR_X        = True
USE_CHINESE     = True

SHOW_GESTURE_TXT= True
SHOW_P9         = False
SHOW_P9_TEXT    = False
SHOW_TIP        = False
SHOW_TIP_TEXT   = False
SHOW_RESET_UI   = True   # ✅ 顯示進度條

# ===== 參數 =====
WINDOW_NAME = "SVM_gesture"
SCALE       = 100
TARGET_ID   = 9

# 食指滑動→角度
ALPHA        = 0.25
DEADZONE_PX  = 1.5
DEG_PER_PX_X = 0.35
DEG_PER_PX_Y = 0.35

# 自動重設（無方框）
RESET_GESTURE_ID  = 2
RESET_HOLD_SEC    = 0.8
RESET_MOVE_THRESH = 2.0

# 顏色
P9_COLOR   = (0, 200, 255)
TIP_COLOR  = (0, 255, 0)

# 類別名稱（依你的模型）
class_names = {0: "張開手", 1: "握拳", 2: "食指"}

# ---------- 中文顯示 ----------
def draw_chinese_text(img, text, pos=(30,30), color=(0,255,0), font_size=36):
    if not USE_CHINESE:
        cv2.putText(img, text.encode("ascii", "ignore").decode(), pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color[::-1], 2)
        return img
    font = None
    for fp in [
        "C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msjh.ttf",
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            continue
    if font is None:
        cv2.putText(img, text.encode("ascii", "ignore").decode(), pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color[::-1], 2)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ---------- 模型 ----------
model = joblib.load('svm_hand_model.pkl')

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
con_spec  = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)

# ---------- 攝影機 ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# ---------- 狀態 ----------
px_s = None
py_s = None
prev_px_s = None
prev_py_s = None
yaw_deg = 0.0
pitch_deg = 0.0
d_yaw = 0.0
d_pitch = 0.0

reset_candidate_start = None
last_reset_x = None
last_reset_y = None

def do_reset():
    """把目前位置設為新原始點，yaw/pitch 歸零。"""
    global yaw_deg, pitch_deg, d_yaw, d_pitch, prev_px_s, px_s, prev_py_s, py_s, reset_candidate_start
    yaw_deg = 0.0
    pitch_deg = 0.0
    d_yaw = 0.0
    d_pitch = 0.0
    if px_s is not None:
        prev_px_s = px_s
    if py_s is not None:
        prev_py_s = py_s
    reset_candidate_start = None

print("q 離開；當手勢=2 且食指穩定停留約 0.8 秒 → 自動把當前位置設為新原始點（yaw/pitch 歸零）。")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if MIRROR_X:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    d_yaw = 0.0
    d_pitch = 0.0
    pred = None
    label_text = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, draw_spec, con_spec)

        # ---- (A) 特徵 ----
        feature = []
        for lm in hand.landmark:
            feature.extend([
                int(round(lm.x * SCALE)),
                int(round(lm.y * SCALE)),
                int(round(lm.z * SCALE)),
            ])

        # ---- (B) SVM 偵測 ----
        if len(feature) == 63:
            pred = int(model.predict([feature])[0])
            label_text = class_names.get(pred, f"Class {pred}")
            if SHOW_GESTURE_TXT:
                frame = draw_chinese_text(frame, f"偵測結果：{label_text}",
                                          pos=(30, 30), color=(0, 255, 0), font_size=40)

        # ---- (C) 計算食指移動 ----
        tip = hand.landmark[8]
        px = tip.x * w
        py = tip.y * h
        if MIRROR_X:
            px = w - px

        if px_s is None:
            px_s, prev_px_s = px, px
        else:
            px_s = ALPHA * px + (1 - ALPHA) * px_s

        if py_s is None:
            py_s, prev_py_s = py, py
        else:
            py_s = ALPHA * py + (1 - ALPHA) * py_s

        dx_px = px_s - prev_px_s
        dy_px = py_s - prev_py_s
        prev_px_s = px_s
        prev_py_s = py_s

        if abs(dx_px) < DEADZONE_PX:
            dx_px = 0.0
        if abs(dy_px) < DEADZONE_PX:
            dy_px = 0.0

        # 左右對應 yaw，上下對應 pitch（往上為正）
        d_yaw = -dx_px * DEG_PER_PX_X
        d_pitch = -dy_px * DEG_PER_PX_Y
        yaw_deg += d_yaw
        pitch_deg += d_pitch

        # ---- (UDP 傳送) ----
        send_data = [int(pred), float(yaw_deg), float(pitch_deg)]
        for lm in hand.landmark:
            x_int = int(w - (lm.x * w)) if MIRROR_X else int(lm.x * w)
            y_int = int(h - (lm.y * h)) if MIRROR_X else int(lm.y * h)
            send_data.extend([x_int, y_int, float(lm.z)])
        sock.sendto(str(send_data).encode(), serverAddressPort)

        # ---- (D) 自動重設與跑條顯示 ----
        gesture_ok = (pred == RESET_GESTURE_ID)
        stable_ok = True
        draw_x = int(w - tip.x * w) if MIRROR_X else int(tip.x * w)
        draw_y = int(tip.y * h)
        if last_reset_x is not None and last_reset_y is not None:
            if abs(draw_x - last_reset_x) > RESET_MOVE_THRESH or abs(draw_y - last_reset_y) > RESET_MOVE_THRESH:
                stable_ok = False
        last_reset_x, last_reset_y = draw_x, draw_y

        if gesture_ok and stable_ok:
            if reset_candidate_start is None:
                reset_candidate_start = time.time()
            hold_elapsed = time.time() - reset_candidate_start
            progress = np.clip(hold_elapsed / RESET_HOLD_SEC, 0.0, 1.0)

            # ✅ 跑條顯示
            if SHOW_RESET_UI:
                frame = draw_chinese_text(frame, "保持食指穩定以重設",
                                          pos=(30, 280), color=(180,180,180), font_size=26)
                bar_w, bar_h = 220, 12
                bar_x, bar_y = 30, 310
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (80,80,80), 1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_w * progress), bar_y + bar_h), (0,220,0), -1)

            if hold_elapsed >= RESET_HOLD_SEC:
                do_reset()
        else:
            reset_candidate_start = None

    # ---- (E) 顯示資訊 ----
    cv2.putText(frame, f"dYaw: {d_yaw:+6.2f} deg", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Yaw : {yaw_deg:+7.1f} deg", (30, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, f"dPitch: {d_pitch:+6.2f} deg", (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Pitch : {pitch_deg:+7.1f} deg", (30, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

