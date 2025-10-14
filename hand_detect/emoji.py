import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

SMILE_THRESHOLD = 0.35
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

try:
    normalPic = cv2.imread("plain.png")
    straight_pic = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    thinkingPic = cv2.imread("pensivemonkey.jpeg")
    thinkingHardPic = cv2.imread("thinking.jpeg")
    if normalPic is None:
        raise FileNotFoundError("plain.png not found")
    if straight_pic is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if thinkingPic is None:
        raise FileNotFoundError("air.jpg not found")
    if thinkingHardPic is None:
        raise FileNotFoundError("thinking.jpeg not found")
    normalPic = cv2.resize(normalPic, EMOJI_WINDOW_SIZE)
    straight_pic = cv2.resize(straight_pic, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    thinkingPic = cv2.resize(thinkingPic, EMOJI_WINDOW_SIZE)
    thinkingHardPic = cv2.resize(thinkingHardPic, EMOJI_WINDOW_SIZE)
    
    print("[.] All emoji images loaded successfully!")
    
except Exception as e:
    print("[!] Error loading emoji images! Make sure they are in the correct folder and named properly.")
    print(f"Error details: {e}")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

print("[.] Starting webcam capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[!] Error: Could not open webcam. Make sure your camera is connected and not being used by another application.")
    exit()

cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Output', WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Output', WINDOW_WIDTH + 150, 100)


def get_mouth_landmark_indices():
    lips_pairs = mp_face_mesh.FACEMESH_LIPS
    indices = set()
    for a, b in lips_pairs:
        indices.add(a)
        indices.add(b)
    return sorted(indices)


def landmarks_to_pixel_coords(landmarks, image_width, image_height):
    pts = []
    for lm in landmarks:
        x_px = int(lm.x * image_width)
        y_px = int(lm.y * image_height)
        pts.append((x_px, y_px))
    return pts


def compute_mouth_center_and_radius(face_landmarks, image_width, image_height):
    mouth_idxs = get_mouth_landmark_indices()
    pts = []
    for idx in mouth_idxs:
        lm = face_landmarks.landmark[idx]
        pts.append((lm.x * image_width, lm.y * image_height))
    if not pts:
        return None, None
    pts = np.array(pts)
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    radius = float(dists.max())
    return (int(center[0]), int(center[1])), max(5, int(radius))


print("[.] Initializing MediaPipe FaceMesh and Hands...")
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
hands = mp.solutions.hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

FINGER_TIP_INDICES = [4, 8, 12, 16, 20]

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[!] Warning: empty frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        thinking_detected = False

        mouth_center = None
        mouth_radius = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            mouth_center, mouth_radius = compute_mouth_center_and_radius(face_landmarks, w, h)
            if mouth_center is not None:
                cv2.circle(frame, mouth_center, mouth_radius, (0, 255, 255), 2)

        if hand_results.multi_hand_landmarks and mouth_center is not None:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                pts = landmarks_to_pixel_coords(hand_landmarks.landmark, w, h)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                face_width = max(1, int(w * 0.25)) if mouth_radius is None else max(1, mouth_radius * 6)
                threshold = max(20, int(mouth_radius * 1.5))

                for idx in FINGER_TIP_INDICES:
                    if idx < len(pts):
                        fx, fy = pts[idx]
                        cv2.circle(frame, (fx, fy), 6, (0, 0, 255), -1)
                        dx = fx - mouth_center[0]
                        dy = fy - mouth_center[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist <= threshold:
                            thinking_detected = True
                            cv2.line(frame, (fx, fy), mouth_center, (0, 255, 0), 2)

        if thinking_detected:
            emoji = thinkingPic.copy()
            label = "Thinking - hand near mouth"
        else:
            emoji = normalPic.copy()
            label = "No hand near mouth"
            
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Output', emoji)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("[.] Interrupted by user")
finally:
    cap.release()
    face_mesh.close()
    hands.close()
    cv2.destroyAllWindows()


