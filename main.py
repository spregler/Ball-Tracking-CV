from tkinter import W
import numpy as np
import cv2 as cv
import scipy.signal as sig
import imutils
import time


def getPerspectiveTransformation(src, dst):
    """ 
    Calculates 3x3 Homography Matrix used to apply a perspective transformation.
    Inputs: src and destination points (4 points of each).
    Outputs: 3 x 3 homography matrix.
    *
    * Coefficients are calculated by solving linear system: Ph = 0
    *
    *     | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | /h11| | 0 |
    *     | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h12| | 0 |
    *     | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h13| | 0 |
    *     | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' |.|h21|=| 0 |
    *     | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h22| | 0 |
    *     | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h23| | 0 |
    *     | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h31| | 0 |
    *     | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h32| | 0 |
    *     | 0   0   0   0   0   0   0   0  1 | | 1 | | 1 |
    *
    *
    """
    if (src.shape != (4,2) or dst.shape != (4, 2) ):
        raise ValueError("There must be four source and destination points")

    p = np.zeros((9, 9))
    px = np.zeros((4, 9))
    py = np.zeros((4, 9))
    b = np.zeros(9)

    for i in range(4):
        px[i][0] = -src[i][0]
        px[i][1] = -src[i][1]
        px[i][2] = -1
        px[i][3] = p[i][4] = p[i][5] = 0
        px[i][6] = src[i][0] * dst[i][0]
        px[i][7] = src[i][1] * dst[i][0]
        px[i][8] = dst[i][0]
        py[i][0] = py[i][1] = py[i][2] = 0
        py[i][3] = -src[i][0]
        py[i][4] = -src[i][1]
        py[i][5] = -1
        py[i][6] = src[i][0] * dst[i][1]
        py[i][7] = src[i][1] * dst[i][1]
        py[i][8] = dst[i][1]

    p[0] = px[0]
    p[1] = py[0]
    p[2] = px[1]
    p[3] = py[1]
    p[4] = px[2]
    p[5] = py[2]
    p[6] = px[3]
    p[7] = py[3]
    p[8] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    b[8] = 1

    h = np.linalg.solve(p, b)
    h.resize((9,), refcheck=False)
    return h.reshape(3, 3)


def click_event(event, x, y, flags, params):
    global counter
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        pts1[counter] = x,y
        counter+=1
        print(pts1)
 

if __name__ == "__main__":
    pts1 = np.zeros((4,2), np.float32)
    counter = 0

    cap = cv.VideoCapture(0)
    cv.namedWindow("test")

    while True:
        isTrue, frame = cap.read()

        if not isTrue:
            print("failed to grab frame")
            break

        cv.imshow("test", frame)
        k = cv.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("ESC Pressed, Closing...")
            break

        elif k%256 == 32:
            img_name = "opencv_frame.png"
            cv.imwrite(img_name, frame)
            print("{} written...".format(img_name))
            key = cv.waitKey(1)
    
    # Read snapshot and find corners
    img = cv.imread(img_name)
    cv.imshow("Original Image ", img)
    cv.setMouseCallback("Original Image ", click_event)


    cv.waitKey(0)
    cv.destroyAllWindows()

    ################################################ Begin Tracking #####################################################################
    cap = cv.VideoCapture(0)
    time.sleep(2.0)

    while True:
        _, frame = cap.read()
        # Crop image to match that of the src image
        #frame_crop = frame[35:440, 150:580] # (height, width)
        # Find Homography transformation 
        pts2 = np.float32([[0, 0], [700, 0], [0, 700], [700, 700]])
        matrix = getPerspectiveTransformation(pts1, pts2)
        # Apply transformation
        warped_frame = cv.warpPerspective(frame, matrix, (700, 700) )

        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)

        blurred = cv.GaussianBlur(warped_frame, (11, 11), 0)
        width, height = warped_frame.shape[:2]
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (0,153,153), (10, 250, 250)) # (0, 0, 255 - sensitivity) - (255, sensitivity, 255)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # To see the centroid clearly
            if radius > 10:
                cv.circle(warped_frame, (int(x), int(y)), int(radius), (0, 255, 255), 5)
                cv.imwrite("circled_frame.png", cv.resize(frame, (int(height / 2), int(width / 2))))
                cv.circle(warped_frame, center, 5, (0, 0, 255), -1)

        cv.imshow("Perspective Transformation", warped_frame)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()