#! /usr/bin/python3
import yaml
from au_functions import ARGS
from remot import REMOT
import cv2 as cv
import numpy as np
import dv_processing as dv
from matplotlib import pyplot as plt
import sys

config_dir = "./config/shape_6dof_fifo.yml"

with open(config_dir, "r") as file:
    config = yaml.safe_load(file)

config = ARGS(config)
remot = REMOT(config)

track_by = "event"
read_cnt = 0
last_frame = None
roi = [None, None]
roi_select_state = 0
roi_hist = None

backSub = cv.createBackgroundSubtractorKNN(20, 100, False)

def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

def getEvents(recording, frame = None):
    global read_cnt
    try:
        events = recording.getNextEventBatch()
    except:
        pass
    read_cnt += 1
    if frame is not None and events is not None:
        for event in events:
            frame[event.y(), event.x()] = (0,0,255) if event.polarity() else (0,255,0)
    
    # TODO: return np mat [[x,y,t,p]]
    return events

def getImage(recording):
    global read_cnt
    try:
        frame = recording.getNextFrame()
    except:
        pass
    if frame is not None:
        read_cnt += 1
        return recording.getNextFrame().image
    else:
        return np.full((260,346,3), 255, 'uint8')

def on_mouse(event, x, y, flags, userdata):
    global roi_select_state, roi, last_frame, roi_hist
    # Left click
    if event == cv.EVENT_LBUTTONUP:
        # Select first point
        if roi_select_state == 0:
            print("select one point")
            roi[0] = (x,y)
            roi_select_state += 1
        # Select second point
        elif roi_select_state == 1:
            roi[1] = (x,y)
            roi_select_state += 1
            roi_p1, roi_p2 = roi
            min_y, max_y = min(roi_p1[1], roi_p2[1]), max(roi_p1[1], roi_p2[1])
            min_x, max_x = min(roi_p1[0], roi_p2[0]), max(roi_p1[0], roi_p2[0])

            if (max_y - min_y) == 0 or (max_x - min_x) == 0:
                print("too close")
                roi_select_state = 1
            else:
                print("roi selected")
                roi_frame = last_frame[min_y: max_y, min_x : max_x]
                hsv_roi =  cv.cvtColor(roi_frame, cv.COLOR_BGR2HSV)
                mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
                cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
                cv.namedWindow('roi_frame', cv.WINDOW_NORMAL)
                cv.resizeWindow('roi_frame', abs(roi_p2[0] - roi_p1[0]), abs(roi_p1[1] - roi_p2[1]))
                cv.imshow('roi_frame', roi_frame)

    # Right click (erase current ROI)
    if event == cv.EVENT_RBUTTONUP:
        print("roi selection cleared")
        roi = [None, None]
        roi_select_state = 0


frame_delay = 200
frame_delay_default = 40
reader = dv.io.MonoCameraRecording(sys.argv[1])
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 346*3, 260*3)
cv.setMouseCallback('frame', on_mouse)
frame = np.full((260,346,3), 255, 'uint8')

for _ in range(19):
    # events = getEvents(reader, frame)
    frame = getImage(reader)

# setup initial window location
# x, y, w, h = 173, 130, 168, 125
track_window = (173, 130, 346, 240)
# term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 8, 5)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
previous_center = np.array([0,0])
track = np.full((260,346,3), 0, 'uint8')
center_path = np.full((260,346,3), 0, 'uint8')
while reader.isRunning():
    # events are formated in the following way: [x, y, self.t, p]
    annotated = np.full((260,346,3), 0, 'uint8')
    if track_by == "event":
        frame = np.full((260,346,3), 0, 'uint8')
        events = getEvents(reader, frame)
        
        read_cnt += 1
        dst = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dst = cv.GaussianBlur(dst,(5,5),0)
        ret, dst = cv.threshold(dst,24,255,cv.THRESH_TOZERO)
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        center = np.int0(ret[0])
        # print("center", center)
        annotated = cv.polylines(frame,[pts],True, (0,255,255), 2)
        cv.putText(annotated, f"x: {center[0]} y:{center[1]} r:{(ret[2]-90):3.2f}", (0, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)

        if not np.all(previous_center == 0) and not np.all(center == 0):
            cv.line(track, previous_center, center, (0,128,0), 1)
            cv.circle(center_path, center, 2, (128, 128, 255), 1)

        previous_center = center
        annotated = cv.add(annotated, track)
        annotated = cv.add(annotated, center_path)
    elif track_by == "image":
        frame = getImage(reader)
        if last_frame is None:
            for _ in range(4):
                frame = getImage(reader)
                backSub.apply(frame)
            last_frame = frame.copy()
        else:
            fgmask = backSub.apply(frame)
            bgmask = cv.bitwise_not(fgmask)
            frame = increase_brightness(frame, 40)
            fg = cv.bitwise_and(frame, frame, mask = fgmask)
            bg = cv.bitwise_and(last_frame, last_frame, mask=bgmask)
            blended = cv.add(fg, bg)
            last_frame = blended.copy()
            annotated = blended

    cv.putText(annotated, f"pkt#: {read_cnt}", (0, 32), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)
    cv.imshow('frame', annotated)
    # cv.imshow('frame', events)
    key = cv.waitKey(frame_delay) & 0xFF
    if key == ord('q'):
        print("read cnt = ", read_cnt)
        exit(0)
    elif key == ord('w'):
        frame_delay  = max(1, frame_delay - 10)
        print("frame delay = ", frame_delay)
    elif key == ord('s'):
        frame_delay  = min(1000, frame_delay + 10)
        print("frame delay = ", frame_delay)
    elif key == ord('d'):
        for _ in range(10):
            if track_by == "event":
                events = getEvents(reader)
            elif track_by == "image":
                img = getImage(reader)
            # read_cnt += 1
        print(read_cnt)

cv.putText(frame, "Press Q to exit", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2, cv.LINE_AA)
cv.imshow('frame', frame)
while cv.waitKey(10) & 0xFF != ord('q'):
    pass
