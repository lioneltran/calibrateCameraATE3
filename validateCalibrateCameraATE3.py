#!/usr/bin/env python
"""
PiCamera Class
Created: 2019 Feb 20
Author: Lionel Tran

Description:
Validate Calibrate Camera on ATE3
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
import readchar
import imutils
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

# Blur
BLURRY_MAX                  = 0

ADJUST_DONE                 = False
ADJUST_STEP                 = 0

ADJUST_XY                   = False

LENSES_TEXT                 = ""
LENSES_TEXT_COLOR           = (0, 0, 255)
XYZ_TEXT                    = ""
XYZ_TEXT_COLOR              = (0, 0, 255)

FRAME_COUNT                 = 0

# for cropping, top-left of 4 points:
LEFT                        = (310 + 36, 90 + 187)
RIGHT                       = (310 + 36, 510 + 187)
TOP                         = (80 + 36, 330 + 187)
BOT                         = (500 + 36, 330 + 187)

LEFT_CORRECT_Y              = 46 #46
RIGHT_CORRECT_Y             = 51 #51
TOP_CORRECT_X               = 46 #46
BOT_CORRECT_X               = 40 #40

LEFT_FLAG                   = False
RIGHT_FLAG                  = False
TOP_FLAG                    = False
BOT_FLAG                    = False

LONGER_EDGE                 = 100
SHORTER_EDGE                = 40

XY_DEVIATION                = 3
BLUR_DEVIATION              = 10
RED                         = (0, 0, 255)
GREEN                       = (0, 255, 0)
ARROW_WEIGHT                = 2

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

def extreme(img, whatExtreme):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, KERNEL3)
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        if whatExtreme == 'left':
            return extLeft
        if whatExtreme == 'right':
            return extRight
        if whatExtreme == 'top':
            return extTop
        if whatExtreme == 'bot':
            return extBot
    else:
        return (0,0)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image_RGB = frame.array
    copy_RGB = image_RGB.copy()
    grey = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

    '''
    Blur detection
    '''
    gray = cv2.cvtColor(copy_RGB, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # print(fm)
    LENSES_TEXT_COLOR = RED
    if fm > BLURRY_MAX and ADJUST_STEP > 0:
        BLURRY_MAX = fm

    if ADJUST_DONE == False:
        if ADJUST_STEP == 0:
            FRAME_COUNT = FRAME_COUNT + 1
            if FRAME_COUNT == 8:
                ADJUST_STEP = 1

        if ADJUST_STEP == 1:
            LENSES_TEXT = "Rotate lens clockwise" 
            if BLURRY_MAX - fm > 70:
                LENSES_TEXT = "Rotate lens counterclockwise" 
                ADJUST_STEP = 2

        if ADJUST_STEP == 2:
            if BLURRY_MAX - fm < 10:
                LENSES_TEXT = "Rotate lens counterclockwise" 
                ADJUST_STEP = 3

        if ADJUST_STEP == 3:
            if BLURRY_MAX - fm > 70:
                LENSES_TEXT = "Rotate lens clockwise"
                ADJUST_STEP = 4

        if ADJUST_STEP == 4:
            LENSES_TEXT = "Rotate lens clockwise slowly"
            if BLURRY_MAX - fm < BLUR_DEVIATION:
                color = (0, 255, 0)
                LENSES_TEXT = "Lens DONE"
                # ADJUST_DONE = True
                ADJUST_STEP = 5

        if ADJUST_STEP == 5:
            if BLURRY_MAX - fm > BLUR_DEVIATION:
                LENSES_TEXT_COLOR = RED
                LENSES_TEXT = "Adjust lens needed"
                XYZ_TEXT = ""

            else:
                XYZ_TEXT_COLOR = RED
                XYZ_TEXT = "Adjusting x, y, z ..."
                LENSES_TEXT_COLOR = GREEN
                LENSES_TEXT = "Lens DONE"

                left_img    = copy_RGB[LEFT[0]:LEFT[0]+SHORTER_EDGE, LEFT[1]:LEFT[1]+LONGER_EDGE]
                right_img   = copy_RGB[RIGHT[0]:RIGHT[0]+SHORTER_EDGE, RIGHT[1]:RIGHT[1]+LONGER_EDGE]
                top_img     = copy_RGB[TOP[0]:TOP[0]+LONGER_EDGE, TOP[1]:TOP[1]+SHORTER_EDGE]
                bot_img     = copy_RGB[BOT[0]:BOT[0]+LONGER_EDGE, BOT[1]:BOT[1]+SHORTER_EDGE]

                left    = (extreme(left_img, 'left')[0], int(SHORTER_EDGE/2))
                right   = (extreme(right_img, 'right')[0], int(SHORTER_EDGE/2))
                top     = (int(SHORTER_EDGE/2) , extreme(top_img, 'top')[1])
                bot     = (int(SHORTER_EDGE/2) , extreme(bot_img, 'bot')[1])

                LEFT_CORRECT    = (LEFT_CORRECT_Y, int(SHORTER_EDGE/2))
                RIGHT_CORRECT   = (RIGHT_CORRECT_Y, int(SHORTER_EDGE/2))
                TOP_CORRECT     = (int(SHORTER_EDGE/2) , TOP_CORRECT_X)
                BOT_CORRECT     = (int(SHORTER_EDGE/2) , BOT_CORRECT_X)

                # print(left, right, top, bot)

                # validate left
                if left[0] < LEFT_CORRECT[0] - XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (left[0] + LEFT[1], left[1] + LEFT[0]), (left[0] + LEFT[1] + 30, LEFT_CORRECT[1] + LEFT[0]), RED, ARROW_WEIGHT)
                    LEFT_FLAG = False
                elif left[0] > LEFT_CORRECT[0] + XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (left[0] + LEFT[1], left[1] + LEFT[0]), (left[0] + LEFT[1] - 30, LEFT_CORRECT[1] + LEFT[0]), RED, ARROW_WEIGHT)
                    LEFT_FLAG = False
                else:
                    cv2.putText(copy_RGB, 'LEFT OK', (LEFT_CORRECT[0] + LEFT[1], LEFT_CORRECT[1] + LEFT[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
                    LEFT_FLAG = True


                # validate right
                if right[0] < RIGHT_CORRECT[0] - XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (right[0] + RIGHT[1], right[1] + RIGHT[0]), (right[0] + RIGHT[1] + 30, RIGHT_CORRECT[1] + RIGHT[0]), RED, ARROW_WEIGHT)
                    RIGHT_FLAG = False
                elif right[0] > RIGHT_CORRECT[0] + XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (right[0] + RIGHT[1], right[1] + RIGHT[0]), (right[0] + RIGHT[1] - 30, RIGHT_CORRECT[1] + RIGHT[0]), RED, ARROW_WEIGHT)
                    RIGHT_FLAG = False
                else:
                    cv2.putText(copy_RGB, 'RIGHT OK', (RIGHT_CORRECT[0] + RIGHT[1], RIGHT_CORRECT[1] + RIGHT[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
                    RIGHT_FLAG = True

                # validate top
                if top[1] < TOP_CORRECT[1] - XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (top[0] + TOP[1], top[1] + TOP[0]), (top[0] + TOP[1], TOP_CORRECT[1] + TOP[0] + 30), RED, ARROW_WEIGHT)
                    TOP_FLAG = False
                elif top[1] > TOP_CORRECT[1] + XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (top[0] + TOP[1], top[1] + TOP[0]), (top[0] + TOP[1], TOP_CORRECT[1] + TOP[0] - 30), RED, ARROW_WEIGHT)
                    TOP_FLAG = False
                else:
                    cv2.putText(copy_RGB, 'TOP OK', (TOP_CORRECT[0] + TOP[1], TOP_CORRECT[1] + TOP[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
                    TOP_FLAG = True

                # validate bot
                if bot[1] < BOT_CORRECT[1] - XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (bot[0] + BOT[1], bot[1] + BOT[0]), (bot[0] + BOT[1], BOT_CORRECT[1] + BOT[0] + 30), RED, ARROW_WEIGHT)
                    BOT_FLAG = False
                elif bot[1] > BOT_CORRECT[1] + XY_DEVIATION:
                    cv2.arrowedLine(copy_RGB, (bot[0] + BOT[1], bot[1] + BOT[0]), (bot[0] + BOT[1], BOT_CORRECT[1] + BOT[0] - 30), RED, ARROW_WEIGHT)
                    BOT_FLAG = False
                else:
                    cv2.putText(copy_RGB, 'BOT OK', (BOT_CORRECT[0] + BOT[1], BOT_CORRECT[1] + BOT[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
                    BOT_FLAG = True

                if LEFT_FLAG == True and RIGHT_FLAG == True and TOP_FLAG == True and BOT_FLAG == True:
                    XYZ_TEXT_COLOR = GREEN
                    XYZ_TEXT = "XYZ DONE"
    # print(ADJUST_STEP)
    # show the image
    # cv2.putText(copy_RGB, "{}: {:.2f}, {:.2f}, {}".format(text, fm, BLURRY_MAX, ADJUST_STEP), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
    # crop_img = copy_RGB[36:705, 187:877]
    # cv2.imwrite("watchface.jpeg", crop_img)
    cv2.putText(copy_RGB, LENSES_TEXT, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LENSES_TEXT_COLOR, 3)
    cv2.putText(copy_RGB, XYZ_TEXT, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, XYZ_TEXT_COLOR, 3)

    cv2.circle(copy_RGB, (x_center, y_center), 216, (0, 0, 255), 1)
    cv2.rectangle(copy_RGB, (187, 36),(877, 705), (0, 0, 255), 2) # (187, 33),(877, 710)

    # cv2.rectangle(copy_RGB, (330 + 187, 100 + 36),(370 + 187, 150 + 36), (0, 0, 255), 2) # up
    # cv2.rectangle(copy_RGB, (330 + 187, 520 + 36),(370 + 187, 570 + 36), (0, 0, 255), 2) # down
    # cv2.rectangle(copy_RGB, (110 + 187, 310 + 36),(160 + 187, 350 + 36), (0, 0, 255), 2) # left
    # cv2.rectangle(copy_RGB, (530 + 187, 310 + 36),(580 + 187, 350 + 36), (0, 0, 255), 2) # right

    # # gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
 
    # # ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
 
    # # im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # # for cnt in contours:
    # #     approx = cv2.approxPolyDP(cnt,2,True) # 2 is 
    # #     print(len(approx))
    # #     if len(approx)==4:
    # #         cv2.drawContours(crop_img,[cnt],0,(0,0,255),-1)
    # #     # else:
    # #     #     cv2.drawContours(crop_img,[cnt],0,(0,255,0),-1)
    # # cv2.rectangle(copy_RGB, (450, 250),(600, 350), (0, 0, 255), 2) # (187, 33),(877, 710)

    cv2.namedWindow("Camera Calibration", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Camera Calibration", copy_RGB)

    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break

