#!/usr/bin/env python
"""
PiCamera Class
Created: 2017 Oct 15
Author: Thien Doan

Description:
Implementation of the Pi Camera class.
"""
import time
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import numpy as np
os.system('sudo modprobe bcm2835-v4l2')


#Camera Params
EXPOSURE_COMPENSATION_WHITE = 1
EXPOSURE_COMPENSATION_BLACK = -8
CAMERA_RESOLUTION_X         = 2592
CAMERA_RESOLUTION_Y         = 1944
CAMERA_SHARPNESS            = 50
CAMERA_CONTRAST             = 50
CAMERA_BRIGHTNESS           = 50
CAMERA_SATURATION           = -30
CAMERA_ISO                  = 0
CAMERA_STABILIZATION        = True
CAMERA_EXPOSURE_MODE        = 'night'
CAMERA_METER_MODE           = 'average'
CAMERA_MODE                 = 'off'
CAMERA_AWB_GAINS            = 1.2
CAMERA_ROTATION             = 0
CAMERA_HFLIP                = False
CAMERA_VFLIP                = False
MAX_DIST_TWO_CENTERS        = 16
CONTOUR_MIN_AREA            = 40
CONTOUR_MAX_AREA            = 1000000
EINK_ROTATE_ANGLE_UPPER     = 16.53  # 15 (constant angle)+1.53 (limit)
EINK_ROTATE_ANGLE_LOWER     = 13.47  # 15 (constant angle)-1.53 (limit)
PCBA_ROTATE_ANGLE_UPPER     = 12.59  # 11.06(constant angle)+1.53(limit)
PCBA_ROTATE_ANGLE_LOWER     = 9.530  # 11.06(constant angle)-1.53(limit)
CHASSIS_HOLES_HORIZ_ANGLE   = 15  # angle between 2 chassis holes and perfect horizontal is 15 degrees (constant)
PCBA_HORIZ_ANGLE            = 11.06  # angle between 2 fiducial holes and perfect horizontal is 15 degrees (constant)



FIDUCIAL_RADIUS_MAX         = 26
FIDUCIAL_RADIUS_MIN         = 22
HOLE_RADIUS_MAX             = 40
HOLE_RADIUS_MIN             = 30
FIDUCIAL_MIN_AREA           = 100
FIDUCIAL_MAX_AREA           = 200

RECT_MIN_AREA               = 100
KERNEL1                     = np.ones((1, 1), np.uint8)
KERNEL2                     = np.ones((2, 2), np.uint8)
KERNEL3                     = np.ones((3, 3), np.uint8)
KERNEL5                     = np.ones((5, 5), np.uint8)
KERNEL7                     = np.ones((7, 7), np.uint8)
KERNEL9                     = np.ones((9, 9), np.uint8)
GAUSSIAN_BLUR_KSIZE_9       = (9, 9)
GAUSSIAN_BLUR_KSIZE_5       = (5, 5)
GAUSSIAN_BLUR_KSIZE_3       = (3, 3)
CANNY_THRESHOLD_1           = 100  # first threshold for the hysteresis procedure.
CANNY_THRESHOLD_2           = 200  # second threshold for the hysteresis procedure.
APERTURE_SIZE               = (2, 2)  # aperture size for the Sobel() operator


h=800
w=1000
x_center = 534
y_center = 370

# Concentricity Test Params
CONCENTRICITY_CROP_X1       = x_center - 14
CONCENTRICITY_CROP_X2       = x_center - 14 + 30
CONCENTRICITY_CROP_Y1       = y_center - 14
CONCENTRICITY_CROP_Y2       = y_center - 14 + 30
PINION_X_CENTER             = 190
PINION_Y_CENTER             = 160
PINION_BINARY_THRESHOLD     = [50, 60, 70]
EINK_BINARY_THRESHOLD       = [40, 50, 60]
EINK_INNER_RADIUS           = 40
EINK_OUTER_RADIUS           = 140
EINK_INNER_RADIUS_MIN       = 65
EINK_INNER_RADIUS_MAX       = 70
INNER_CIRCLE_RADIUS_MIN     = 18
INNER_CIRCLE_RADIUS_MAX     = 23
OUTER_CIRCLE_RADIUS_MIN     = 67
OUTER_CIRCLE_RADIUS_MAX     = 70
PINION_MIN_AREA             = 50
PINION_MAX_AREA             = 200
PINION_RADIUS_MASK          = 45
EINK_MIN_AREA               = 12000
EINK_MAX_AREA               = 13500
# Alignment Test Params
# Rectangle cropped positions
RECT_THRES_BINARY           = 180
ALIGN_RECT_CROP_X_1         = 1000
ALIGN_RECT_CROP_X_2         = 1600
ALIGN_RECT_CROP_Y_1         = 400
ALIGN_RECT_CROP_Y_2         = 800

# Fiducial and Hole cropped positions
FIDUCIAL_THRES_BINARY       = [140, 180, 200]
HOLE_THRES_BINARY           = [30, 40, 50, 60, 70, 80, 90, 20]
ALIGN_F1_CROP_X_1           = 95
ALIGN_F1_CROP_X_2           = 155
ALIGN_F1_CROP_Y_1           = 420
ALIGN_F1_CROP_Y_2           = 480

ALIGN_F2_CROP_X_1           = 870
ALIGN_F2_CROP_X_2           = 930
ALIGN_F2_CROP_Y_1           = 290
ALIGN_F2_CROP_Y_2           = 350


ALIGN_H1_CROP_X_1           = 110
ALIGN_H1_CROP_X_2           = 160
ALIGN_H1_CROP_Y_1           = 270
ALIGN_H1_CROP_Y_2           = 350

ALIGN_H2_CROP_X_1           = 870
ALIGN_H2_CROP_X_2           = 915
ALIGN_H2_CROP_Y_1           = 490
ALIGN_H2_CROP_Y_2           = 540
HOLE_MIN_AREA               = 50
HOLE_MAX_AREA               = 400

camera = PiCamera()
camera.resolution = (w, h)
camera.exposure_mode = CAMERA_EXPOSURE_MODE
camera.contrast = CAMERA_CONTRAST
camera.brightness = CAMERA_BRIGHTNESS
camera.saturation = CAMERA_SATURATION
camera.ISO = CAMERA_ISO
camera.video_stabilization = CAMERA_STABILIZATION
camera.exposure_mode = CAMERA_EXPOSURE_MODE
camera.sharpness = CAMERA_SHARPNESS
camera.meter_mode = CAMERA_METER_MODE
camera.awb_mode = CAMERA_MODE
camera.awb_gains = CAMERA_AWB_GAINS
camera.image_effect = 'none'
camera.color_effects = None
camera.rotation = CAMERA_ROTATION
camera.hflip = CAMERA_HFLIP
camera.vflip = CAMERA_VFLIP
camera.crop = (0.0, 0.0, 1.0, 1.0)
camera.exposure_compensation = EXPOSURE_COMPENSATION_WHITE
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(w, h))
time.sleep(2)
camera.capture('/home/pi/Desktop/calib.jpeg')

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image_RGB = frame.array
    copy_RGB = image_RGB.copy()
    grey = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)


    cv2.circle(copy_RGB, (x_center, y_center), 216, (0, 0, 255), 1)
    
##    cv2.circle(copy_RGB, (x_center, y_center), 5, (0, 0, 255), 5)

##    cv2.rectangle(copy_RGB, (ALIGN_F1_CROP_X_1-5, ALIGN_F1_CROP_Y_1-5), (ALIGN_F1_CROP_X_1+60, ALIGN_F1_CROP_Y_1+60), (0, 0, 255), 1)
##    cv2.rectangle(copy_RGB, (ALIGN_F2_CROP_X_1-5, ALIGN_F2_CROP_Y_1-5), (ALIGN_F2_CROP_X_1+60, ALIGN_F2_CROP_Y_1+60), (0, 0, 255), 1)
##    cv2.rectangle(copy_RGB, (ALIGN_H1_CROP_X_1-5, ALIGN_H1_CROP_Y_1-5), (ALIGN_H1_CROP_X_1+60, ALIGN_H1_CROP_Y_1+60), (0, 0, 255), 1)
##    cv2.rectangle(copy_RGB, (ALIGN_H2_CROP_X_1-5, ALIGN_H2_CROP_Y_1-5), (ALIGN_H2_CROP_X_1+60, ALIGN_H2_CROP_Y_1+60), (0, 0, 255), 1)
    cv2.rectangle(copy_RGB, (187, 33),(877, 710), (0, 0, 255), 2)
##    cv2.imwrite('img_calibrateCamera.jpeg', copy_RGB)

##    # copy_RGB = cv2.imread('/home/pi/Desktop/calib.jpeg')
##    '''
##    Fiducial detection
##    '''
##    f1 = copy_RGB[ALIGN_F1_CROP_Y_1:ALIGN_F1_CROP_Y_2, ALIGN_F1_CROP_X_1:ALIGN_F1_CROP_X_2]
##    cv2.imwrite('/home/pi/Desktop/F1.jpeg', f1)
##    f2 = copy_RGB[ALIGN_F2_CROP_Y_1:ALIGN_F2_CROP_Y_2, ALIGN_F2_CROP_X_1:ALIGN_F2_CROP_X_2]
##    cv2.imwrite('/home/pi/Desktop/F2.jpeg', f2)
##
##    image_blur = cv2.GaussianBlur(f1, GAUSSIAN_BLUR_KSIZE_3, 0)
##    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
##    for threshold in FIDUCIAL_THRES_BINARY:
##        ret, image_binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
##        # Enhance edge of circle
##        image_binary = cv2.dilate(image_binary, KERNEL2, iterations=1)
##        image_binary = cv2.erode(image_binary, KERNEL2, iterations=1)
##        dept, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
##        # Find the index of the largest contour
##        areas = [cv2.contourArea(c) for c in contours]
##        max_index = 0
##        best_area = 0
##        for i in range(0, len(areas)):
##            print(areas[i])
##            if areas[i] > FIDUCIAL_MIN_AREA and areas[i] < FIDUCIAL_MAX_AREA:
##                cnt = contours[i]
##                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
##                # #Only get ellipse ~ circle to remove edge of E-Ink
##                if ma / MA < 1.7:
##                    max_index = i
##                    best_area = areas[i]
##        if best_area != 0:
##            # max_index = np.argmax(areas)
##            cnt = contours[max_index]
##            ellipse = cv2.fitEllipse(cnt)
##            overlap_ellipse = ((ellipse[0][0]+ALIGN_F1_CROP_X_1, ellipse[0][1]+ALIGN_F1_CROP_Y_1), (ellipse[1][0], ellipse[1][1]), ellipse[2])
##            cv2.ellipse(copy_RGB, overlap_ellipse, (0, 255, 0), 2)
##
##    '''
##    Fiducial 2
##    '''
##
##    image_blur = cv2.GaussianBlur(f2, GAUSSIAN_BLUR_KSIZE_3, 0)
##    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
##    for threshold in FIDUCIAL_THRES_BINARY:
##        ret, image_binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
##        # Enhance edge of circle
##        image_binary = cv2.dilate(image_binary, KERNEL2, iterations=1)
##        image_binary = cv2.erode(image_binary, KERNEL2, iterations=1)
##        dept, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
##        # Find the index of the largest contour
##        areas = [cv2.contourArea(c) for c in contours]
##        max_index = 0
##        best_area = 0
##        for i in range(0, len(areas)):
##            print(areas[i])
##            if areas[i] > FIDUCIAL_MIN_AREA and areas[i] < FIDUCIAL_MAX_AREA:
##                cnt = contours[i]
##                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
##                # #Only get ellipse ~ circle to remove edge of E-Ink
##                if ma / MA < 1.7:
##                    max_index = i
##                    best_area = areas[i]
##        if best_area != 0:
##            # max_index = np.argmax(areas)
##            cnt = contours[max_index]
##            ellipse = cv2.fitEllipse(cnt)
##            overlap_ellipse = (
##            (ellipse[0][0] + ALIGN_F2_CROP_X_1, ellipse[0][1] + ALIGN_F2_CROP_Y_1), (ellipse[1][0], ellipse[1][1]),ellipse[2])
##            cv2.ellipse(copy_RGB, overlap_ellipse, (0, 255, 0), 2)
##
##
##    '''
##    Pinion detection
##    '''
##    p = copy_RGB[CONCENTRICITY_CROP_Y1:CONCENTRICITY_CROP_Y2, CONCENTRICITY_CROP_X1:CONCENTRICITY_CROP_X2]
##    cv2.imwrite('/home/pi/Desktop/P.jpeg', p)
##    image_blur = cv2.GaussianBlur(p, GAUSSIAN_BLUR_KSIZE_3, 0)
##    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
##    for threshold in PINION_BINARY_THRESHOLD:
##        ret, image_binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
##        # Enhance edge of circle
##        image_binary = cv2.dilate(image_binary, KERNEL2, iterations=1)
##        image_binary = cv2.erode(image_binary, KERNEL2, iterations=1)
##        dept, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
##        # Find the index of the largest contour
##        areas = [cv2.contourArea(c) for c in contours]
##        max_index = 0
##        best_area = 0
##        for i in range(0, len(areas)):
##            print(areas[i])
##            if areas[i] > PINION_MIN_AREA and areas[i] < PINION_MAX_AREA:
##                cnt = contours[i]
##                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
##                # #Only get ellipse ~ circle to remove edge of E-Ink
##                if ma / MA < 1.5:
##                    max_index = i
##                    best_area = areas[i]
##        if best_area != 0:
##            # max_index = np.argmax(areas)
##            cnt = contours[max_index]
##            ellipse = cv2.fitEllipse(cnt)
##            overlap_ellipse = ((ellipse[0][0] + CONCENTRICITY_CROP_X1, ellipse[0][1] + CONCENTRICITY_CROP_Y1), (ellipse[1][0], ellipse[1][1]),ellipse[2])
##            cv2.ellipse(copy_RGB, overlap_ellipse, (0, 255, 0), 1)

    '''
    Chassis hole
    '''
    # h1 = copy_RGB[ALIGN_H1_CROP_Y_1:ALIGN_H1_CROP_Y_2, ALIGN_H1_CROP_X_1:ALIGN_H1_CROP_X_2]
    # cv2.imwrite('/home/pi/Desktop/H1.jpeg', h1)
    # image_blur = cv2.GaussianBlur(h1, GAUSSIAN_BLUR_KSIZE_3, 0)
    # image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    # for threshold in HOLE_THRES_BINARY:
    #     ret, image_binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
    #     # Enhance edge of circle
    #     image_binary = cv2.dilate(image_binary, KERNEL2, iterations=1)
    #     image_binary = cv2.erode(image_binary, KERNEL2, iterations=1)
    #     dept, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #     # Find the index of the largest contour
    #     areas = [cv2.contourArea(c) for c in contours]
    #     max_index = 0
    #     best_area = 0
    #     for i in range(0, len(areas)):
    #         print(areas[i])
    #         if areas[i] > HOLE_MIN_AREA and areas[i] < HOLE_MAX_AREA:
    #             cnt = contours[i]
    #             (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    #             # #Only get ellipse ~ circle to remove edge of E-Ink
    #             if ma / MA < 1.7:
    #                 max_index = i
    #                 best_area = areas[i]
    #     if best_area != 0:
    #         # max_index = np.argmax(areas)
    #         cnt = contours[max_index]
    #         ellipse = cv2.fitEllipse(cnt)
    #         overlap_ellipse = ((ellipse[0][0] + ALIGN_H1_CROP_X_1, ellipse[0][1] + ALIGN_H1_CROP_Y_1),
    #                            (ellipse[1][0], ellipse[1][1]), ellipse[2])
    #         cv2.ellipse(copy_RGB, overlap_ellipse, (0, 255, 0), 2)
    #
    #
    # h2 = copy_RGB[ALIGN_H2_CROP_Y_1:ALIGN_H2_CROP_Y_2, ALIGN_H2_CROP_X_1:ALIGN_H2_CROP_X_2]
    # cv2.imwrite('/home/pi/Desktop/H2.jpeg', h2)
    # image_blur = cv2.GaussianBlur(h1, GAUSSIAN_BLUR_KSIZE_3, 0)
    # image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    # for threshold in HOLE_THRES_BINARY:
    #     ret, image_binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
    #     # Enhance edge of circle
    #     image_binary = cv2.dilate(image_binary, KERNEL2, iterations=1)
    #     image_binary = cv2.erode(image_binary, KERNEL2, iterations=1)
    #     dept, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #     # Find the index of the largest contour
    #     areas = [cv2.contourArea(c) for c in contours]
    #     max_index = 0
    #     best_area = 0
    #     for i in range(0, len(areas)):
    #         print(areas[i])
    #         if areas[i] > HOLE_MIN_AREA and areas[i] < HOLE_MAX_AREA:
    #             cnt = contours[i]
    #             (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    #             # #Only get ellipse ~ circle to remove edge of E-Ink
    #             if ma / MA < 1.5:
    #                 max_index = i
    #                 best_area = areas[i]
    #     if best_area != 0:
    #         # max_index = np.argmax(areas)
    #         cnt = contours[max_index]
    #         ellipse = cv2.fitEllipse(cnt)
    #         overlap_ellipse = ((ellipse[0][0] + ALIGN_H2_CROP_X_1, ellipse[0][1] + ALIGN_H2_CROP_Y_1),(ellipse[1][0], ellipse[1][1]), ellipse[2])
    #         cv2.ellipse(copy_RGB, overlap_ellipse, (0, 255, 0), 2)

    cv2.imshow("Camera Calibration", copy_RGB)

    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break

