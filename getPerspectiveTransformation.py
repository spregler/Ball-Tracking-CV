from tkinter import W
import numpy as np
import cv2 as cv
import scipy.signal as sig

def getPerspectiveTransformation(src, dst):
    """ 
    Calculates 3x3 Homography Matrix to transform the src points to the dst points
    *
    * Coefficients are calculated by solving linear system:
    *             P                          H  = 0
    * / -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' \ /h11\ / 0 \
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h12| | 0 |
    * | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h13| | 0 |
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' |.|h21|=| 0 |,
    * | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h22| | 0 |
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h23| | 0 |
    * | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h31| | 0 |
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h32| | 0 |
    * \ 0   0   0   0   0   0   0   0  1 / \(1)/ \ 1 /
    *
    */
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


def harris(img, threshold=0.79):
    # Sobel x-axis kernel
    SOBEL_X = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int32")

    # Sobel y-axis kernel
    SOBEL_Y = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int32")

    # Gaussian kernel
    GAUSS = np.array((
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]), dtype="float32")

    img_crop = img[35:440, 150:580] # img[row, col]
    img_cpy = img_crop.copy()
    img_gray = cv.cvtColor(img_crop, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5,5), 1)
    # Convolve w/ Sobel operator on both axis
    dx = sig.convolve2d(img_gray, SOBEL_X)
    dy = sig.convolve2d(img_gray, SOBEL_Y)
    # Square of derivatives
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxdy = dx * dy
    # Convolve w/ Gaussian kernel
    dx2_g = sig.convolve2d(dx2, GAUSS)
    dy2_g = sig.convolve2d(dy2, GAUSS)
    dxdy_g = sig.convolve2d(dxdy, GAUSS)
    # Calculate score: R = det(M) - k*trace^2(M)
    R = dx2_g*dy2_g - np.square(dxdy_g) - 0.05*np.square(dx2_g + dy2_g)
    # Normalize
    cv.normalize(R, R, 0, 1, cv.NORM_MINMAX)
    # Nonmax supression
    sup = np.where(R >= threshold) # sup[0] is x-axis and sup[1] is y-axis

    for px in zip(*sup[::-1]):
        cv.circle(img_cpy, px, 2, (0, 0, 255), -1)

    return img_cpy, sup



if __name__ == "__main__":
    # Create a transform to change table coordinates in inches to projector coordinates
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
    corners, sup = harris(img)

    # DEBUG - Output x and y positions of corners
    print(sup[0]) # y-position
    print(sup[1]) # x-position

    pts = []

    # If there are four points detected, append to pts1
    if (len(sup[0]) == 4):
        for i in range(4):
                pts.append([sup[1][i], sup[0][i]])
        print(pts)
    # There are more than four corners detected, we must remove corners that are being detected twice
    elif (len(sup[0]) > 4):
        for i in range(len(sup[0]) - 1):
            if ( (sup[0][i+1] - sup[0][i] <= 1) & (sup[1][i+1] - sup[1][i] <= 1) ):
                print("detected same point")
            else: # ( (sup[0][i+1] - sup[0][i]) > 1 or (sup[1][i+1] - sup[1][i]) > 1 ):
                pts.append([sup[1][i], sup[0][i]])
        pts.append([sup[1][len(sup[0])-1], sup[0][len(sup[0])-1]]) # add last element of the array


    print("Detected Points: ", pts)
    pts1 = np.array(pts, dtype=np.float32)


    cv.imshow("Result", corners)

    cv.waitKey(0)
    cv.destroyAllWindows()

    ################################################ Begin Tracking #####################################################################
    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
        matrix = getPerspectiveTransformation(pts1, pts2)

        result = cv.warpPerspective(frame, matrix, (400, 400) )
        cv.imshow("Perspective Transformation", result)

        key = cv.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    # while True:
    #     isTrue, frame = cap.read()

    #     if not isTrue:
    #         print("failed to grab frame")
    #         break

    #     cv.imshow("tracking", frame)

    #     if cv.waitKey(1) == 27:
    #         break


    