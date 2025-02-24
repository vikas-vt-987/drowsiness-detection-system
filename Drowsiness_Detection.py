import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
from playsound import playsound
import imutils
import threading

# Load Dlib models
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Constants
thresh = 0.25
closed_eye_time_threshold = 0.50
closed_eye_timer = 0.0
alert_triggered = False
playing_sound = False

# Eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Function to calculate Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to play alert sound
def play_alert_sound():
    global playing_sound
    if not playing_sound:
        playing_sound = True
        playsound("audio/beep-warning-6387.mp3")
        playing_sound = False

# Start Video Capture
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30.0  # Default FPS if not available

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Check if the webcam is connected.")
        break
    
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    eyes_closed = False

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            eyes_closed = True

    if eyes_closed:
        closed_eye_timer += 1 / fps
        if closed_eye_timer >= closed_eye_time_threshold and not alert_triggered:
            print("Drowsy!")
            threading.Thread(target=play_alert_sound, daemon=True).start()
            alert_triggered = True
    else:
        closed_eye_timer = 0
        alert_triggered = False

    flipped_frame = cv2.flip(frame, 1)
    cv2.imshow("Frame", flipped_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
