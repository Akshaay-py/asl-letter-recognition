import cv2

# Open the default webcam (0 = built-in camera)
cap = cv2.VideoCapture(0)

# Check that the camera opened properly
if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
else:
    print("Webcam opened. Press Q on the video window to quit.")
    while True:
        # Read one picture (frame) from the camera
        success, frame = cap.read()
        if not success:
            print("ERROR: Could not read from the webcam.")
            break

        # Flip the picture left-right so it works like a mirror
        frame = cv2.flip(frame, 1)

        # Show the picture in a window
        cv2.imshow("Webcam Test", frame)

        # Wait 1 millisecond for a key press. Stop if it is Q.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup: release the camera and close the window
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Webcam closed.")
