import cv2
import numpy as np
import time

# print(cv2.__version__)


capture_video = cv2.VideoCapture(0)

# give the camera to warm up
time.sleep(1)
count = 0
background = 0

# capturing the background in range of 60
# you should have video that have some seconds
# dedicated to background frame so that it
# could easily save the background image
for i in range(60):
    return_val, background = capture_video.read()
    if return_val == False:
        continue

    # background = background #np.flip(background,axis=1) # flipping of the frame

# we are reading from video
while (capture_video.isOpened()):
    return_val, img = capture_video.read()
    if not return_val:
        break
    count = count + 1
    # img = img #np.flip(img, axis = 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # -------------------------------------BLOCK----------------------------#
    # range for orange

    # lower_orange = np.array([7, 185, 167])   # harry
    lower_orange = np.array([5, 75, 75])  # invisible man
    # lower_orange = np.array([10, 75, 75])
    upper_orange = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower_orange, upper_orange)
    # cv2.imshow('mask 1',mask1)

    lower_white = np.array([3, 0, 225])
    upper_white = np.array([15, 32, 255])
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    # cv2.imshow('mask 2',mask2)

    mask1 = mask1 + mask2

    # ----------------------------------------------------------------------#

    # Refining the mask corresponding to the detected red color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3),
                                                            np.uint8), iterations=2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow('mask 1 _ morph', mask1)

    mask2 = cv2.bitwise_not(mask1)
    cv2.imshow('mask 2_not', mask2)
    # cv2.waitKey(0)

    # Generating the final output
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("The invisible man", final_output)
    k = cv2.waitKey(10)
    if k == 27:
        break

cv2.waitKey(0)