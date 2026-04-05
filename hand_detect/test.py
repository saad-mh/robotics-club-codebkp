from math import e

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2
)

cap = cv2.VideoCapture(0)

FINGERTIPS = [4, 8, 12, 16, 20]

def fingers_in_rect(fingertips, rect_pts):
    """
    fingertips: list of (x,y)
    rect_pts: numpy array of polygon points
    returns: True if any finger is inside
    """

    for finger in fingertips:
        if cv2.pointPolygonTest(rect_pts, finger, False) >= 0:
            return True

    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    rgb.flags.writeable = True

    h, w, _ = frame.shape

    # ---------------- HAND DETECTION ----------------
    fingertip_positions = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):

                if id in FINGERTIPS:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    fingertip_positions.append((x,y))

                    cv2.circle(frame, (x, y), 6, (0,255,0), -1)


    # ---------------- FACE DETECTION ----------------
    pos_67 = None
    pos_338 = None

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):

                if id in [67, 338]:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    if id == 67:
                        pos_67 = (x, y)
                    elif id == 338:
                        pos_338 = (x, y)

                    cv2.circle(frame, (x, y), 4, (255,0,0), -1)
                    cv2.putText(frame, str(id), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    # draw forehead line if both visible

    if pos_67 and pos_338:
        p1 = np.array(pos_67)
        p2 = np.array(pos_338)

        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length != 0:
            unit_dir = direction / length

            # p vector
            perp = np.array([-unit_dir[1], unit_dir[0]])
            if perp[1] > 0:
                perp = -perp

            extend = 120
            height = 120

            start = p1 - unit_dir * extend
            end = p2 + unit_dir * extend

            start_top = start + perp * height
            end_top = end + perp * height

            pts = np.array([
                start.astype(int),
                end.astype(int),
                end_top.astype(int),
                start_top.astype(int)
            ])

            cv2.polylines(frame, [pts], True, (255,0,0), 2)
            touching = fingers_in_rect(fingertip_positions, pts)
            if touching:
                cv2.putText(frame, "HOLYMOLY", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("???", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()