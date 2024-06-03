import cv2
import numpy as np
import mediapipe as mp
from screeninfo import get_monitors

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.1,
                       min_tracking_confidence=0.1)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Read in M 
M = np.load("M.npy")

camera_matrix = np.load("./camera_matrix.npy")
dist_coeffs = np.load('./dist_coeffs.npy')

width, height = 1920, 1200

# Get monitor information
monitors = get_monitors()

# Choose the second monitor (index 1), or adjust accordingly
if len(monitors) > 1:
    second_monitor = monitors[1]
else:
    second_monitor = monitors[0]  # Fallback to the main monitor if only one monitor is found

while True:
    ret, frame = cap.read()

    # Undistort the frame with camera calibration
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Corrected line
    warped_image = cv2.warpPerspective(frame, M, (width, height))

    # Convert to RGB
    rgb_frame = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    # Run inference for hand detection
    results = hands.process(rgb_frame)

    warped_image = np.zeros((width, height, 3), np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(warped_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Rotate the image 180
    warped_image = cv2.rotate(warped_image, cv2.ROTATE_180)

    # Create a named window
    cv2.namedWindow("Final Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Final Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Move window to the second monitor
    cv2.moveWindow("Final Image", second_monitor.x, second_monitor.y)
    
    cv2.imshow("Final Image", warped_image)
    cv2.waitKey(1)
