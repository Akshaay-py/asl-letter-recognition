import os
import cv2
import joblib
import mediapipe as mp
from collections import Counter, deque

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_FOLDER, "sign_model.joblib")

PROJECT_TITLE = "ASL Letter Recognition"
MIN_CONFIDENCE = 0.8    # at least 4 of the 5 neighbours must agree
MAX_DISTANCE = 0.8      # use the same value that worked for you in Step 6
SMOOTHING_FRAMES = 8    # the shown answer is the most common of the last 8
SHOW_DEBUG = False      # change to True to see confidence and distance

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_features(hand_landmarks):
    """Turn the 21 hand points into 42 numbers (same as in Step 4)."""
    wrist = hand_landmarks.landmark[0]
    numbers = []
    for point in hand_landmarks.landmark:
        numbers.append(point.x - wrist.x)
        numbers.append(point.y - wrist.y)
    biggest = max(abs(n) for n in numbers)
    if biggest == 0:
        biggest = 1
    return [n / biggest for n in numbers]


def predict_letter(model, hand_landmarks):
    """Return (answer, confidence, distance). The answer is a letter
    or the word Unknown."""
    features = get_features(hand_landmarks)
    probabilities = model.predict_proba([features])[0]
    best = probabilities.argmax()
    letter = str(model.classes_[best])
    confidence = probabilities[best]
    distances, _ = model.kneighbors([features])
    avg_distance = distances[0].mean()

    if confidence >= MIN_CONFIDENCE and avg_distance <= MAX_DISTANCE:
        return letter, confidence, avg_distance
    return "Unknown", confidence, avg_distance


def main():
    if not os.path.exists(MODEL_FILE):
        print("ERROR: sign_model.joblib not found. Run step5_train.py first.")
        return

    model = joblib.load(MODEL_FILE)
    supported = " ".join(str(c) for c in model.classes_)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
        print("Close other apps using the camera and check the camera permission.")
        return

    print("Started. Press Q on the video window to quit.")
    recent = deque(maxlen=SMOOTHING_FRAMES)
    details = ""

    try:
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    answer, confidence, distance = predict_letter(
                        model, hand_landmarks
                    )
                    details = "confidence " + str(round(confidence, 2)) \
                              + "  distance " + str(round(distance, 2))
                else:
                    answer = "None"
                    details = ""

                # Show the most common answer of the last few frames
                recent.append(answer)
                shown = Counter(recent).most_common(1)[0][0]

                if shown == "None":
                    message = "No hand detected"
                    color = (0, 0, 255)        # red
                elif shown == "Unknown":
                    message = "Unknown gesture"
                    color = (0, 165, 255)      # orange
                else:
                    message = "Letter: " + shown
                    color = (0, 255, 0)        # green

                # Dark see-through strip at the top and bottom for the text
                height, width = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
                cv2.rectangle(overlay, (0, height - 35), (width, height),
                              (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

                cv2.putText(frame, PROJECT_TITLE, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, message, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
                cv2.putText(frame, "Supported: " + supported + "   |   Q = quit",
                            (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1)
                if SHOW_DEBUG and details:
                    cv2.putText(frame, details, (width - 330, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

                cv2.imshow(PROJECT_TITLE, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # This always runs, even after an error, so the camera is released
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Webcam closed.")


main()