import dv_processing as dv
from multiprocessing import Queue
from scipy import io
import numpy as np
import sys
import time
import csv

reader_queue = Queue(100)
reader_source = sys.argv[1]

csv_file = open('pkt.log.csv', 'w+')
pkt_log = csv.writer(csv_file)
pkt_log.writerow(['packet_cnt', 'packet_drop', 'queue size'])

def event_to_array_and_frame(events):
    ts = 0
    if events is not None:
        frame = np.full((260,346,3), 0, 'uint8')
        event_array = np.ndarray((len(events), 3), dtype=np.uint32)
        event_idx = 0
        for event in events:
            if frame is not None:
                frame[event.y(), event.x()] = (0,0,230) if event.polarity() else (0,230,0)
            event_array[event_idx] = np.array([event.x(), event.y(), event.timestamp() & 0xFFFFFFFF])
            event_idx += 1
            ts = event.timestamp()
    else:
        event_array = None
    return (event_array, frame, ts)
   
def DAVIS_Reader_Process():
    global reader_queue, reader_source
    print("Started Reader Process")
    if reader_source == 'camera':
        reader = dv.io.CameraCapture()
    elif reader_source == 'shapes':
        shapes_event = io.loadmat('dataset/shapes.mat')['events']

    else:
        reader = dv.io.MonoCameraRecording(reader_source)

    event_ts = 0
    image_ts = 0
    print("Start reading")

    if reader_source == 'shapes':
        frame_count = 1
        tFrame = 40000
        t = shapes_event[:, 3]
        total_time = t[-1]
        for i in range(total_time // (1 * tFrame)):
            idxs = np.where((t > frame_count * tFrame) & (t <= (frame_count + 1) * 1 * tFrame))[0]
            events = shapes_event[idxs]
            frame = np.full((260,346,3), 0, 'uint8')
            for event in events:
                frame[event[1], event[0]] = (0,0,230) if event[3] else (0,230,0)
            ts = events[-1][2]
            frame_count += 1
            reader_result = ((events, frame, ts), None)
            reader_queue.put(reader_result)
    else:
        packet_cnt = 0
        drop_cnt = 0

        while reader.isRunning():
            events = reader.getNextEventBatch()
            packet_cnt += 1
            if packet_cnt == 1300:
                break
            events = event_to_array_and_frame(events)
            event_ts = events[2]
            if event_ts > image_ts:
                frame = reader.getNextFrame()
                image_ts = frame.timestamp
                image = frame.image
            else:
                image = None
            reader_result = (events, image)
            
            if reader_queue.full():
                drop_cnt += 1
                print(f"Buffer full! dropped {drop_cnt} packets")
            else:
                reader_queue.put(reader_result)
            
            # print("Queue size", reader_queue.qsize())
            if reader_source != "camera":
                time.sleep(0.01)

            pkt_log.writerow([packet_cnt, drop_cnt, reader_queue.qsize()])

        csv_file.close()
        print("Reader process terminated")
    return 0