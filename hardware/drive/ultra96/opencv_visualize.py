#! /usr/bin/python3
import yaml
from au_functions import ARGS
from remot import REMOT
from trajectory import Trajectory
from davis_reader import DAVIS_Reader_Process
import cProfile
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import time
import csv
# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool, Queue, cpu_count

perf_log = csv.writer(open('perf.log.csv', 'w+'))
perf_log.writerow(['packet_cnt', 'event_cnt', 'object_cnt', 'process_rate', 'power'])

track_log = csv.writer(open('track.log.csv', 'w+'))
track_log.writerow(['packet_cnt', 'object id', 'AU id', 'x', 'y', 'r'])

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

if len(sys.argv) >= 3:
    headless = int(sys.argv[2])
else:
    headless = 0

if not headless:
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', 346*3, 260*3)

event_pkt_cnt = 0
image_frame_cnt = 0
backSub = cv.createBackgroundSubtractorKNN(20, 100, False)
frame_delay = 1
trajectory = []
image_last_updated = -1
last_render = time.time()
last_frame = None
remot_prof = cProfile.Profile()

def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

def event_to_frame(frame, events, color):
    y = events[:, 1]
    x = events[:, 0]
    frame[x, y] = color
    return frame


# for _ in range(980):
#     event_batch = reader.getNextEventBatch()
#     events = getEvents()

from davis_reader import reader_queue
original_image_frame = None
annotated_image_frame = None

with Pool(cpu_count() - 1) as process_pool:
    remot = REMOT(config, process_pool)
    events_read_result = None
    event_ts = 0
    event_cnt = 0
    backsub_init_cnt = 4

    davis_reader_result = process_pool.apply_async(DAVIS_Reader_Process)
    end_of_stream = False

    for i in range(980):
        event_pkt_cnt += 1
        reader_queue.get()

    while not reader_queue.full():
        pass

    while not davis_reader_result.ready() or not reader_queue.empty():
        if event_pkt_cnt == 1200:
            break
        #######################################
        # event process
        #######################################
        # original_event_frame = np.full((260,346,3), 0, 'uint8')
        annotated_event_frame = np.full((260,346,3), 0, 'uint8')
        # events are formated in the following way: [x, y, self.t, p]
        # remot_prof.enable()
        (event_result, image) = reader_queue.get()
        event_pkt_cnt += 1

        (events, original_event_frame, ts) = event_result

        event_ts = events[-1, 2]
        event_cnt = events.shape[0]
        # remot_prof.enable()
        live_au, tracking_state, au_fifo = remot.update(events)
        # remot_prof.create_stats()
        # remot_prof.dump_stats('remot.prof')

        for au_id in live_au:
            tracking_id, tracking_ts = tracking_state[au_id]
            events = au_fifo[au_id]

            if not headless:
                event_to_frame(annotated_event_frame, au_fifo[au_id], cmap[tracking_id % len(cmap)])

            if tracking_id > len(trajectory) - 1:
                # print("Add new tracker")
                trajectory.append(Trajectory(tracking_id))
            result = trajectory[tracking_id].update(events)
            track_log.writerow([event_pkt_cnt, tracking_id, au_id, result[0][0], result[0][1], result[2]])

            # print(f'AU {au_id} tracking object {tracking_id} result: {result}')
            
        if not headless:
            for t in trajectory:
                t.draw(annotated_event_frame)

        #######################################
        # image process
        #######################################
        if not headless and image is not None:
            original_image_frame = image

            if backsub_init_cnt:
                backSub.apply(original_image_frame)
                backsub_init_cnt -= 1
                if backsub_init_cnt == 0:
                    last_frame = original_image_frame.copy()
            else:
                annotated_image_frame = original_image_frame.copy()
                fgmask = backSub.apply(annotated_image_frame)
                bgmask = cv.bitwise_not(fgmask)
                # annotated_image_frame = increase_brightness(annotated_image_frame, 40)
                fg = cv.bitwise_and(annotated_image_frame, annotated_image_frame, mask = fgmask)
                bg = cv.bitwise_and(last_frame, last_frame, mask=bgmask)
                blended = cv.add(fg, bg)
                last_frame = blended.copy()
                annotated_image_frame = blended
                cv.putText(original_image_frame, "Original image frame", (80, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)
                cv.putText(annotated_image_frame, "Annotated image frame", (80, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)

        if image is None and original_image_frame is None:
            if last_frame is None:
                original_image_frame = np.full((260,346,3), 0, 'uint8')
            else:
                original_image_frame = last_frame
        
        if annotated_image_frame is None:
            annotated_image_frame = original_image_frame

        current_time = time.time()
        update_rate = 1.0 / (current_time - last_render)
        last_render = current_time

        print(f'event packet count: {event_pkt_cnt}')
        print(f'update rate: {update_rate}')
        # perf_log.writerow([event_pkt_cnt, events.shape[0], len(live_au), update_rate, remot.get_power()])
        perf_log.writerow([event_pkt_cnt, event_cnt, len(live_au), update_rate, reader_queue.qsize()])


        if not headless:
            cv.putText(original_event_frame, "Original event packet", (80, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)
            cv.putText(annotated_event_frame, "Annotated event frame", (80, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)

            combined_event_frame = np.concatenate((original_event_frame, annotated_event_frame), axis = 0)
            combined_image_frame = np.concatenate((original_image_frame, annotated_image_frame), axis = 0)
            combined_frame = np.concatenate((combined_event_frame, combined_image_frame), axis = 1)

            cv.putText(combined_frame, f"event pkt #: {event_pkt_cnt}", (0, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)
            cv.putText(combined_frame, f"image frame #: {image_frame_cnt}", (0, 64), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)

            cv.putText(combined_frame, f"Process rate: {update_rate:2f}", (0, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv.LINE_AA)
            cv.imshow('frame', combined_frame)

            key = cv.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                print(f'event pkt cnt = {event_pkt_cnt} image frame cnt = {image_frame_cnt}')
                exit(0)
            elif key == ord('w'):
                frame_delay  = max(1, frame_delay - 10)
                print("frame delay = ", frame_delay)
            elif key == ord('s'):
                frame_delay  = min(1000, frame_delay + 10)
                print("frame delay = ", frame_delay)
            elif key == ord('d'):
                for _ in range(100):
                    events = getEvents(reader)
                print(f'event pkt cnt = {event_pkt_cnt}')
            elif key == ord('c'):
                print("Clear previous tracking")
                for t in trajectory:
                    t.clear()
                (original_image_frame, image_last_updated) = getImage(reader)
                backSub.apply(original_image_frame)
                last_frame = original_image_frame

        # remot_prof.create_stats()
        # remot_prof.dump_stats('remot.prof')

    if not headless:
        frame = np.full((260,346,3), 255, 'uint8')
        cv.putText(frame, "Press Q to exit", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2, cv.LINE_AA)
        cv.imshow('frame', frame)
        while cv.waitKey(10) & 0xFF != ord('q'):
            pass
