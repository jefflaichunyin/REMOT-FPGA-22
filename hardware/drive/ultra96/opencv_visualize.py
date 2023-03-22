#! /usr/bin/python3
import yaml
from au_functions import ARGS
from remot import REMOT
from trajectory import Trajectory

import cv2 as cv
import numpy as np
import dv_processing as dv
from matplotlib import pyplot as plt
import sys

config_dir = "./config/shape_6dof_fifo.yml"

cmap = [
    [0,0,255],      # red
    [0,128,255],    # tangerin
    [0,191,255],    # orange
    [0,255,255],    # yellow
    [0,255,191],    # light green
    [0,255,0],      # green
    [191,255,0],    # lake green
    [255,255,0],    # sky blue
    [255,191,0],    # navy blue
    [255,0,0],      # deep blue
    [255,0,128],    # purple
    [255,0,255]     # magenta
]

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
    if events is not None:
        event_array = np.ndarray((len(events), 4), dtype=np.uint64)
        event_idx = 0
        for event in events:
            if frame is not None:
                frame[event.y(), event.x()] = (0,0,128) if event.polarity() else (0,128,0)
            event_array[event_idx] = np.array([event.y(), event.x(), event.timestamp() & 0xFFFFFFFF, event.polarity()])
            event_idx += 1
    else:
        event_array = None
    return event_array

def event_to_frame(frame, events, color):
    y = events[:, 1]
    x = events[:, 0]
    frame[x, y] = color
    return frame

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

# def on_mouse(event, x, y, flags, userdata):
#     global roi_select_state, roi, last_frame, roi_hist
#     # Left click
#     if event == cv.EVENT_LBUTTONUP:
#         # Select first point
#         if roi_select_state == 0:
#             print("select one point")
#             roi[0] = (x,y)
#             roi_select_state += 1
#         # Select second point
#         elif roi_select_state == 1:
#             roi[1] = (x,y)
#             roi_select_state += 1
#             roi_p1, roi_p2 = roi
#             min_y, max_y = min(roi_p1[1], roi_p2[1]), max(roi_p1[1], roi_p2[1])
#             min_x, max_x = min(roi_p1[0], roi_p2[0]), max(roi_p1[0], roi_p2[0])

#             if (max_y - min_y) == 0 or (max_x - min_x) == 0:
#                 print("too close")
#                 roi_select_state = 1
#             else:
#                 print("roi selected")
#                 roi_frame = last_frame[min_y: max_y, min_x : max_x]
#                 hsv_roi =  cv.cvtColor(roi_frame, cv.COLOR_BGR2HSV)
#                 mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#                 roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
#                 cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
#                 cv.namedWindow('roi_frame', cv.WINDOW_NORMAL)
#                 cv.resizeWindow('roi_frame', abs(roi_p2[0] - roi_p1[0]), abs(roi_p1[1] - roi_p2[1]))
#                 cv.imshow('roi_frame', roi_frame)

#     # Right click (erase current ROI)
#     if event == cv.EVENT_RBUTTONUP:
#         print("roi selection cleared")
#         roi = [None, None]
#         roi_select_state = 0


frame_delay = 200
frame_delay_default = 40
reader = dv.io.MonoCameraRecording(sys.argv[1])
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 346*3, 260*3)
# cv.setMouseCallback('frame', on_mouse)
frame = np.full((260,346,3), 255, 'uint8')

# for _ in range(19):
#     # events = getEvents(reader, frame)
#     frame = getImage(reader)

trajectory = []
debug_frame = np.full((260,346,3), 0, 'uint8')
while reader.isRunning():
    # events are formated in the following way: [x, y, self.t, p]
    annotated = np.full((260,346,3), 0, 'uint8')
    original_frame = None
    if track_by == "event":
        original_frame = np.full((260,346,3), 0, 'uint8')
        annotated = np.full((260,346,3), 0, 'uint8')
        events = getEvents(reader, original_frame)

        print("\nREMOT Process:")
        live_au, tracking_state, au_fifo = remot.Process(events, True)
        
        if len(live_au):
            print("result:")
        else:
            print("Not tracking")

        for au_id in live_au:
            tracking_id, tracking_ts = tracking_state[au_id]
            events = au_fifo[au_id]
            print(f'AU {au_id} tracking object {tracking_id} ts: {tracking_ts}')
            event_to_frame(annotated, au_fifo[au_id], cmap[tracking_id])

            if tracking_id > len(trajectory) - 1:
                print("Add new tracker")
                trajectory.append(Trajectory(tracking_id))
            trajectory[tracking_id].update(events)
        
        for t in trajectory:
            t.draw(annotated)

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

    combined = np.concatenate((original_frame, annotated), axis = 0)
    cv.putText(combined, f"pkt#: {read_cnt}", (0, 32), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)
    cv.imshow('frame', combined)
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
