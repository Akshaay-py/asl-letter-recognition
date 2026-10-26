import os
import cv2
import joblib
import mediapipe as mp

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_FOLDER, "sign_model.joblib")

# How sure the model must be before we trust it
MIN_CONFIDENCE = 0.8    # at least 4 of the 5 neighbours must agree
MAX_DISTANCE = 0.8      # the hand shape must be close to the examples

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_features(hand_landmarks):
    """Same function as in step4_collect.py: 21 points -> 42 numbers."""
    wrist = hand_landmarks.landmark[0]
    numbers = []
    for point in hand_landmarks.landmark:
        numbers.append(point.x - wrist.x)
        numbers.append(point.y - wrist.y)
    biggest = max(abs(n) for n in numbers)
    if biggest == 0:
        biggest = 1
    return [n / biggest for n in numbers]


if not os.path.exists(MODEL_FILE):
    print("ERROR: sign_model.joblib not found. Run step5_train.py first.")
else:
    model = joblib.load(MODEL_FILE)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
    else:
        print("Live recognition started. Press Q on the video window to quit.")

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

                message = "No hand detected"
                color = (0, 0, 255)       # red
                details = ""

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        features = get_features(hand_landmarks)

                        # Ask the model: which letter, and how sure are you?
                        probabilities = model.predict_proba([features])[0]
                        best = probabilities.argmax()
                        letter = model.classes_[best]
                        confidence = probabilities[best]

                        # How far is this hand from the 5 closest examples?
                        distances, _ = model.kneighbors([features])
                        avg_distance = distances[0].mean()

                        if confidence >= MIN_CONFIDENCE and avg_distance <= MAX_DISTANCE:
                            message = "Letter: " + str(letter)
                            color = (0, 255, 0)       # green
                        else:
                            message = "Unknown gesture"
                            color = (0, 165, 255)     # orange

                        details = "confidence " + str(round(confidence, 2)) \
                                  + "  distance " + str(round(avg_distance, 2))

                cv2.putText(frame, "ASL Letter Recognition", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, message, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
                cv2.putText(frame, details, (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("ASL Letter Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Webcam closed.")
