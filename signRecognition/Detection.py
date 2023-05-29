import cv2
import numpy as np
import time
import imutils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
camera = cv2.VideoCapture(0)


def detectEyes():
    sec = 4
    now = time.time()
    freq = np.array([4])
    while True:
        if sec > time.time() - now:
            # Grab the current frame
            (grabbed, frame) = camera.read()

            if not grabbed:
                print("No input image")
                break

            frame = imutils.resize(frame, width=500)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Draw rectangles around the detected faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract the region of interest (ROI) for the face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detect eyes in the face ROI
                eyes = eye_cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangles around the detected eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    # Determine the center point of each eye
                    eye_center_x = x + ex + ew // 2
                    eye_center_y = y + ey + eh // 2

                    # Perform the desired action based on the eye position
                    if eye_center_x < frame.shape[1] // 3:
                        print("Look left")
                        # Perform action for looking left
                    elif eye_center_x > frame.shape[1] * 2 // 3:
                        print("Look right")
                        # Perform action for looking right
                    else:
                        print("Look straight")
                        # Perform action for looking straight

            # Show the frame with rectangles
            cv2.imshow("Frame", frame)

            # If the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Stop program and close all windows")
                break
        else:
            # Perform the action based on the most frequent eye direction
            counts = np.bincount(freq)
            most_freq = np.argmax(counts)
            if most_freq == 1:
                print("Look left")
                # Perform action for looking left
            elif most_freq == 2:
                print("Look right")
                # Perform action for looking right
            elif most_freq == 3:
                print("Look straight")
                # Perform action for looking straight
            elif most_freq == 4:
                print("No eye detected")
            now = time.time()
            freq = np.array([4])
    return ()

def main():
    detectEyes()


if __name__ == '__main__':
    main()
