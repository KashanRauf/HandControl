import time
import pyautogui as auto
import cv2 as cv
from mediapipe import solutions as solutions


# INITIALIZING OBJECTS/VARIABLES

# Monitor info
width = auto.size()[0]
height = auto.size()[1]
print(width, height, sep=", ")

# Time between frames
pTime = 0
cTime = 0
dTime = 0

# Video capture
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Capture failed, exiting.")
    exit()

capWidth = capture.get(cv.CAP_PROP_FRAME_WIDTH)
capHeight = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
print(capWidth, capHeight, sep=", ")

wRatio = width / capWidth
hRatio = height / capWidth
print(wRatio, hRatio, sep=", ")

# Hands
mpHands = solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = solutions.drawing_utils
prevPos = []
curPos = []

# Main process after initializing everything
while True:
    success, img = capture.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)   
    
    # Updates time, dTime is difference of time
    cTime = time.time()
    dTime = cTime - pTime
    pTime = cTime

    # Get location of wrist, compare to a previous location and time difference
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(hand.landmark):
            h, w, c = img.shape
            if id == 0:
                # Flipping X for consistency with hand and mouse
                curPos = [int(lm.x*w) * -1, int(lm.y*h)]
                distanceAway = lm.z
                break
        # Draws hand on image (only used when displaying camera feed)
        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
        
        if prevPos:
            dx = curPos[0] - prevPos[0]
            dy = curPos[1] - prevPos[1]

            # 12 is just a constant chosen for no particular reason
            mult = dTime * 12
            # Increases multiplier if hand is a certain distance from camera
            if distanceAway < pow(10, -7):
                mult *= 5
            auto.moveRel((dx*wRatio)*mult, (dy*hRatio)*mult)

    # Updates previous position
    prevPos = curPos

    # Display
    cv.imshow("Feed", img)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()