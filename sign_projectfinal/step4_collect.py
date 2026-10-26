import os
import csv
import cv2
import mediapipe as mp

LETTERS = ["a", "b", "i", "l", "v", "y"]
SAMPLES_PER_LETTER = 100

# The data file is saved in the same folder as this program
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "hand_data.csv")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_features(hand_landmarks):
    """Turn the 21 hand points into 42 numbers that do not depend on
    where the hand is on the screen or how far it is from the camera."""
    wrist = hand_landmarks.landmark[0]
    numbers = []
    for point in hand_landmarks.landmark:
        numbers.append(point.x - wrist.x)
        numbers.append(point.y - wrist.y)
    biggest = max(abs(n) for n in numbers)
    if biggest == 0:
        biggest = 1
    return [n / biggest for n in numbers]


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
else:
    print("Data collection started.")
    print("Press a, b, i, l, v or y to record that letter. Press Q to quit.")

    recording_letter = None
    recorded = 0

    # Open the data file in "append" mode so old data is never erased
    with open(DATA_FILE, "a", newline="") as file:
        writer = csv.writer(file)

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
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                        # If we are recording, save this hand shape
                        if recording_letter is not None:
                            row = [recording_letter.upper()]
                            row = row + get_features(hand_landmarks)
                            writer.writerow(row)
                            recorded += 1
                            if recorded >= SAMPLES_PER_LETTER:
                                print("Saved", recorded, "samples for",
                                      recording_letter.upper())
                                recording_letter = None

                if recording_letter is not None:
                    message = "Recording " + recording_letter.upper() + ": " \
                              + str(recorded) + "/" + str(SAMPLES_PER_LETTER)
                else:
                    message = "Press a b i l v y to record. Q = quit"

                cv2.putText(frame, "Sign Language Data Collection", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, message, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Collect Data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if recording_letter is None and key != 255:
                    if chr(key) in LETTERS:
                        recording_letter = chr(key)
                        recorded = 0

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Webcam closed. Data saved in:", DATA_FILE)