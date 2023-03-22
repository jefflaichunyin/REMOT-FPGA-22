import cv2 as cv
import numpy as np

class Trajectory:

    def __init__(self, tracking_id) -> None:
        self.tracking_id = tracking_id
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        self.trajectory = []
        self.track_window = (173, 130, 346, 240)
        self.alive = False

    def event_to_frame(self, events, color):
        frame = np.full((260,346,3), 0, 'uint8')
        y = events[:, 1]
        x = events[:, 0]
        frame[x, y] = color
        return frame

    def update(self, events):
        frame = self.event_to_frame(events, [255,255,255])
        dst = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dst = cv.GaussianBlur(dst,(5,5),0)
        ret, dst = cv.threshold(dst,24,255,cv.THRESH_TOZERO)
        rotated_rect, self.track_window = cv.CamShift(dst, self.track_window, self.term_crit)
        center = np.int0(rotated_rect[0])
        # print(f'tracker {self.tracking_id} center: {center}')
        if not np.all(center == 0):
            self.alive = True
            self.trajectory.append(rotated_rect)

    def draw(self, frame):
        if len(self.trajectory) == 0:
            return
        
        # draw bounding box for latest position
        # cv.putText(annotated, f"x: {center[0]} y:{center[1]} r:{(ret[2]-90):3.2f}", (0, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)
        if self.alive:
            box_points = cv.boxPoints(self.trajectory[-1])
            box_points = np.int0(box_points)
            cv.polylines(frame,[box_points],True, (0,255,255), 2)
        self.alive = False
        # print(f'tracker {self.tracking_id} traj: {self.trajectory}')
        # connect tracking points
        previous_center = None
        for point in self.trajectory:
            center = np.int0(point[0])
            # draw tangent
            tangent_box = list(point)
            tangent_box[1] = (10, 1)
            box_points = cv.boxPoints(tangent_box)
            box_points = np.int0(box_points)
            cv.line(frame, box_points[0], box_points[2], (0,230,80), 1)
            # cv.polylines(frame,[box_points],True, (0,190,190), 2)

            if self.trajectory.index(point) == 0:
                # first point
                previous_center = center
                cv.circle(frame, center, 2, (128, 128, 255), 1)
                continue

            cv.line(frame, previous_center, center, (0,128,0), 1)
            cv.circle(frame, center, 2, (128, 128, 255), 1)

            previous_center = center