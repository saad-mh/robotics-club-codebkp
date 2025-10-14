import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Finger tip and middle joint landmark indices for each finger
FINGER_TIPS = [4, 8, 12, 16, 20]  # Fingers: Thumb, Index, Middle, Ring, Chota finger i forgot the name
FINGER_PIPS = [3, 6, 10, 14, 18]  # Joint points to compare with tips

def count_extended_fingers(hand_landmarks):
    """
    Count how many fingers are extended (not curled up)
    Returns the count of extended fingers
    """
    landmarks = hand_landmarks.landmark
    extended_fingers = 0
    
    # Check each finger
    for i, (tip_id, pip_id) in enumerate(zip(FINGER_TIPS, FINGER_PIPS)):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        
        # For thumb (index 0), check x-coordinate (horizontal extension)
        if i == 0:
            # Thumb is extended if tip is further from wrist than mid joint
            # Use landmark 0 (wrist) as reference point
            wrist = landmarks[0]
            tip_to_wrist_dist = abs(tip.x - wrist.x)
            pip_to_wrist_dist = abs(pip.x - wrist.x)
            if tip_to_wrist_dist > pip_to_wrist_dist:
                extended_fingers += 1
        else:
            # For other fingers, check y-coordinate (vertical extension)
            # Finger is extended if tip is above (lower y value) than mid joint
            if tip.y < pip.y:
                extended_fingers += 1
    
    return extended_fingers

def detect_hands_and_fingers(frame):
    """
    Detect hands using MediaPipe and count extended fingers
    Returns: (finger_count, hand_landmarks_list, processed_frame)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    total_fingers = 0
    hand_landmarks_list = []
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_landmarks_list.append(hand_landmarks)
            fingers = count_extended_fingers(hand_landmarks)
            total_fingers += fingers
    
    return total_fingers, hand_landmarks_list, results

def main():
    """
    Main function to run hand point detection
    """
    print("[.] Starting MediaPipe hand detection...")
    print("[.] Instructions:")
    print("    - Show your hand(s) to the camera")
    print("    - Extend fingers to see them counted")
    print("    - Press 'q' or ESC to quit")
    print("    - Press 's' to save current frame")
    
    # Initialize camera with 0 meaning the default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[!] Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30) # change 30 to: your camera's max FPS if PC is performant, else 30 for performance; suggested val: 60
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands and fingers
            finger_count, hand_landmarks_list, results = detect_hands_and_fingers(frame)
            
            # Draw on the original frame
            display_frame = frame.copy()
            
            # Draw hand landmarks and connections
            if hand_landmarks_list and results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(hand_landmarks_list):
                    # Get hand classification (l/r)
                    hand_label = "Unknown"
                    if results.multi_handedness and i < len(results.multi_handedness):
                        hand_label = results.multi_handedness[i].classification[0].label
                    
                    # Count fingers for this specific hand
                    hand_finger_count = count_extended_fingers(hand_landmarks)
                    
                    # Draw hand skeleton
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Draw fingertips with special highlighting
                    for tip_id in FINGER_TIPS:
                        landmark = hand_landmarks.landmark[tip_id]
                        h, w, _ = display_frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(display_frame, (cx, cy), 8, (255, 255, 0), -1)
                        # cv2.circle(display_frame, (cx, cy), 12, (0, 0, 255), 2) # --- IGNORE ---
                    
                    # Get hand center for text positioning
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = display_frame.shape
                    hand_x, hand_y = int(wrist.x * w), int(wrist.y * h)
                    
                    # Determine hand status
                    hand_status = "Open" if hand_finger_count == 5 else f"{hand_finger_count}/5"
                    
                    # Draw hand label and status
                    label_text = f"{hand_label}: {hand_status}"
                    cv2.putText(display_frame, label_text, (hand_x - 50, hand_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, label_text, (hand_x - 50, hand_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Add info text
            cv2.putText(display_frame, f"Extended Fingers: {finger_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Hands Detected: {len(hand_landmarks_list)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display result
            cv2.imshow('Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Save frame
                filename = f"hand_detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[.] Saved frame as {filename}")
            
            frame_count += 1
            
            # Print finger count every 30 frames; do NOT change this without if you dont know what youre doing.
            if frame_count % 30 == 0 and finger_count > 0:
                print(f"[.] Detected {finger_count} extended fingers on {len(hand_landmarks_list)} hands")
    
    except KeyboardInterrupt:
        print("[.] Interrupted by user")
    
    finally:
        # Remember to release resources kids!
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        print("[.] Hand point detection stopped")

main()