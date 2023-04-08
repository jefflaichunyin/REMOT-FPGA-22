import dv_processing as dv
from multiprocessing import Queue
import numpy as np
import sys

reader_queue = Queue(10)
reader_source = sys.argv[1]

def event_to_array_and_frame(events):
    ts = 0
    if events is not None:
        frame = np.full((260,346,3), 0, 'uint8')
        event_array = np.ndarray((len(events), 3), dtype=np.uint32)
        event_idx = 0
        for event in events:
            if frame is not None:
                frame[event.y(), event.x()] = (0,0,230) if event.polarity() else (0,230,0)
            event_array[event_idx] = np.array([event.y(), event.x(), event.timestamp() & 0xFFFFFFFF])
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
    else:
        reader = dv.io.MonoCameraRecording(reader_source)

    event_ts = 0
    image_ts = 0
    print("Start reading")
    while reader.isRunning():
        events = reader.getNextEventBatch()
        events = event_to_array_and_frame(events)
        event_ts = events[2]
        if event_ts > image_ts:
            frame = reader.getNextFrame()
            image_ts = frame.timestamp
            image = frame.image
        else:
            image = None
        reader_result = (events, image)
        reader_queue.put(reader_result)

    return 0