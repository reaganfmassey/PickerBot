# Python code for Multiple Color Detection


import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

red_lower1 = np.array([0, 100, 20], np.uint8)
red_upper1 = np.array([10, 255, 255], np.uint8)
red_lower2 = np.array([160, 100, 20], np.uint8)
red_upper2 = np.array([170, 255, 255], np.uint8)
green_lower = np.array([36, 52, 70], np.uint8)
green_upper = np.array([80, 255, 255], np.uint8)
blue_lower = np.array([94, 80, 70], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)
yellow_lower = np.array([20, 59, 70], np.uint8)
yellow_upper = np.array([35, 255, 255], np.uint8)

# Start a while loop
while (1):

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_mask1 = cv2.inRange(hsvFrame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    # Set range for green color and
    # define mask
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=blue_mask)
    # For red color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=yellow_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)


    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]
        if area > 400:
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 0), 2)
            if len(approx) == 3:
                cv2.putText(imageFrame, "Red Triangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif len(approx) == 4:
                cv2.putText(imageFrame, "Red Rectangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif 10 < len(approx) < 20:
                cv2.putText(imageFrame, "Red Circle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]
        if area > 400:
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 0), 2)
            if len(approx) == 3:
                cv2.putText(imageFrame, "Green Triangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif len(approx) == 4:
                cv2.putText(imageFrame, "Green Rectangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif 10 < len(approx) < 20:
                cv2.putText(imageFrame, "Green Circle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]
        if area > 400:
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 0), 2)
            if len(approx) == 3:
                cv2.putText(imageFrame, "Blue Triangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif len(approx) == 4:
                cv2.putText(imageFrame, "Blue Rectangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif 10 < len(approx) < 20:
                cv2.putText(imageFrame, "Blue Circle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]
        if area > 400:
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 0), 2)
            if len(approx) == 3:
                cv2.putText(imageFrame, "Yellow Triangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif len(approx) == 4:
                cv2.putText(imageFrame, "Yellow Rectangle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif 10 < len(approx) < 20:
                cv2.putText(imageFrame, "Yellow Circle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
