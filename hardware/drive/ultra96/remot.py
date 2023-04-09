from au_hardware_fifo_only import Au_fifo
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import directed_hausdorff
import pynq
from multiprocessing import *
from remot_process import *
from queue import Queue

class REMOT():
    def __init__(self, args, process_pool):
        self.input = args.input
        self.bitfile = args.bitfile
        self.args = args
        self.auFifo = args.auFifo
        self.auNum = args.auNum
        self.global_id = 0

        self.rails = pynq.get_rails()
        print("Bitfile = ", args.bitfile)
        self.AUs = Au_fifo(Height=args.ly, Width=args.lx, bitfile=args.bitfile, au_number=args.auNum, fifo_depth=args.auFifo)

        self.pkt_cnt = 0
        self.process_pool = process_pool
        self.process_interval = args.psProcessInverval
        self.process_result = Queue(maxsize=cpu_count()-2)
        self.live_au_list = np.where(self.AUs.status_reg==0)[0]

    def get_power(self):
        total = 0.0
        for k in self.rails:
            r = self.rails[k]
            if hasattr(r.power, "value"):
                total += r.power.value
        return total

    def update(self, events):
        self.pkt_cnt += 1
        ts = events[-1, 2]
        self.AUs.stream_in_events(events)
        self.AUs.dump_all_au()

        # if self.pkt_cnt % self.process_interval == 0 and not self.process_result.full():
        #     self.process_result.put(self.process_pool.apply_async(REMOT_Process, (ts, self.AUs.status_reg, self.AUs.au_event_fifo, self.AUs.auNumber, self.global_id, self.args)))

        # if not self.process_result.empty() and self.process_result.queue[0].ready():
        # # if not self.process_result.empty():
        #     (status, fifo, number, global_id) = self.process_result.get().get()
        #     self.AUs.status_reg[:] = status
        #     # print("process status", status)
        #     for i in range(self.args.auNum):
        #         self.AUs.au_event_fifo[i] = fifo[i]
        #         self.AUs.auNumber[i] = number[i]
        #         self.global_id = global_id
        #     self.live_au_list = np.where(self.AUs.status_reg==0)[0]
        #     self.AUs.sync_all_au()
        # else:
        #     # self.global_id = REMOT_Update(ts, self.AUs.status_reg, self.AUs.au_event_fifo, self.AUs.auNumber, self.global_id, self.args)
        #     self.live_au_list = np.where(self.AUs.status_reg==0)[0]

        (status, fifo, number, global_id) = REMOT_Process(ts, self.AUs.status_reg, self.AUs.au_event_fifo, self.AUs.auNumber, self.global_id, self.args)
        self.global_id = global_id
        self.AUs.sync_all_au()
        self.live_au_list = np.where(self.AUs.status_reg==0)[0]

        return (self.live_au_list, self.AUs.auNumber, self.AUs.au_event_fifo)