import cv2
import mediapipe as mp

# Tools from MediaPipe: the hand finder and a helper that draws on the picture
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
else:
    print("Hand detection started. Press Q on the video window to quit.")

    # Create the hand finder. We only look for one hand.
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("ERROR: Could not read from the webcam.")
                break

            frame = cv2.flip(frame, 1)

            # MediaPipe needs RGB colours, but OpenCV gives BGR, so we convert
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Ask MediaPipe to look for a hand in this picture
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                message = "Hand detected"
                # Draw the 21 points and the lines joining them
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            else:
                message = "No hand detected"

            cv2.putText(frame, message, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Webcam closed.")