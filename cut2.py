import csv
import cv2
import mediapipe as mp


def export_hand_landmarks_csv(
        output_path='hand_landmark_integer.csv',
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
):
    # MediaPipe Hands 初始化
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    mp_draw = mp.solutions.drawing_utils
    draw_spec_lms = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
    draw_spec_con = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # 建立 CSV 檔案並寫入表頭
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # 建立標頭：frame, x0, y0, z0, ..., x20, y20, z20
        header = ['frame'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

        frame_idx = 0

        try:
            while True:
                ret, img = cap.read()
                if not ret:
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if result.multi_hand_landmarks:
                    hand_lms = result.multi_hand_landmarks[0]

                    # 繪製
                    mp_draw.draw_landmarks(
                        img, hand_lms, mp_hands.HAND_CONNECTIONS,
                        draw_spec_lms, draw_spec_con
                    )

                    # 建立整數化 landmark 資料
                    row = [frame_idx]
                    for lm in hand_lms.landmark:
                        x_int = int(round(lm.x * 100))
                        y_int = int(round(lm.y * 100))
                        z_int = int(round(lm.z * 100))
                        row.extend([x_int, y_int, z_int])

                    writer.writerow(row)

                cv2.imshow('Hand Tracking', img)
                frame_idx += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            print(f"Done. Integer CSV saved to {output_path}")


if __name__ == '__main__':
    export_hand_landmarks_csv()
