from pynq import Overlay 
from pynq import allocate
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

class Au_full:
    def __init__(self, Height, Width, bitfile, au_number=16, fifo_depth=2048, dAdd=3):
        self.Height = Height
        self.Width = Width
        self.overlay = Overlay(bitfile)
        self.au = self.overlay.top_0
        self.overlay.download()
        self.heatmap_precision = np.uint16
        self.event_precision = np.uint64
        self.au_number = au_number
        self.fifo_depth = fifo_depth
        self.key_width = 16
        self.dAdd = dAdd
        
        self.event_buffer = allocate(shape=(8192,), dtype=self.event_precision)
        self.output_buffer = allocate(shape=(Height * Width,), dtype=self.heatmap_precision)
        self.init_buffer = allocate(shape=(Height * Width,), dtype=self.heatmap_precision)
        self.in_fifo = allocate(shape=(self.fifo_depth, ), dtype=self.event_precision)
        self.out_fifo = allocate(shape=(self.fifo_depth, ), dtype=self.event_precision)
        
        self.output_buffer_address = 0x10
        self.init_buffer_address = 0x1C
        self.out_fifo_address = 0x28
        self.init_fifo_address = 0x34
        
        self.N_event_address = 0x40
        self.retrive_n_address = 0x48
        self.init_n_address = 0x50
        self.init_fifo_depth_address = 0x58
        self.empty_in = 0x60
        self.empty_out = 0x68
        self.xbits = 10
        self.ybits = 10
        self.tbits = 32
        self.pbits = 1
        
        self.au.write(self.output_buffer_address, self.output_buffer.physical_address)
        self.au.write(self.init_buffer_address, self.init_buffer.physical_address)
        self.au.write(self.out_fifo_address, self.out_fifo.physical_address)
        self.au.write(self.init_fifo_address, self.in_fifo.physical_address)
        self.send = self.overlay.axi_dma_0.sendchannel
        
        self.amap = np.zeros([self.au_number, self.Height, self.Width], dtype=np.uint16)
        self.au_event_fifo = [np.zeros([self.fifo_depth, 4], dtype=np.uint32) for i in range(self.au_number)] 
        self.status_reg = np.ones([self.au_number], dtype=np.uint8)
        self.total_time = 0
        self.auBox = [[0,0,0,0] for i in range(self.au_number)]
        self.auNumber = [[0, 0] for i in range(self.au_number)]
        self.init_hardware()
        

    def init_hardware(self):
        pack = 0
        for i in range(self.au_number):
            pack += (1 << i)
        pack = int(pack)
        self.au.write(self.empty_in, pack)
        self.run_empty()

    def write_status(self):
        # print(self.status_reg)
        pack = 0
        for i in range(self.au_number):
            pack += (self.status_reg[i]<<i)

        pack = int(pack)
        self.au.write(self.empty_in, pack)

    def run_empty(self):
        self.au.write(self.init_n_address, 100) # a large number larger than au number
        self.au.write(self.N_event_address, 0)
        self.au.write(self.retrive_n_address, 100)
        self.au.write(0x00, 0x01)
        while not (self.au.read(0x0) & 0x2):
            pass
        
    def read_status(self, update=False):
        if update:
            self.run_empty()
        pack = self.au.read(self.empty_out)
        for i in range(self.au_number):
            self.status_reg[i] = (pack >> i) & 1

    def pack_event(self, events):
        event = events.astype(np.uint64)
        x = event[:, 0]
        y = event[:, 1]
        t = event[:, 2]
        p = event[:, 3]
        pack = x + (y << self.xbits) + (t << (self.xbits + self.ybits)) + (p << (self.xbits + self.ybits + self.tbits))      
        return pack  

    def unpack_event(self, packed_events):
        x = packed_events & 0x3FF
        y = (packed_events >> self.xbits) & 0x3FF
        t = (packed_events >> (self.xbits + self.ybits)) & 0xFFFFFFFF
        p = (packed_events >> (self.xbits + self.ybits + self.tbits)) & 0x1
        events = np.vstack([x, y, t, p]).T  
        return events      

    def allocate_event_buffer(self, N):
        self.event_buffer = allocate(shape=(N), dtype=self.event_precision)

    def write_au(self, event, number):
        event = event[-self.fifo_depth:]
        number = int(number)
        amap = self.rebuild_amap_with_event(event)
        self.amap[number] = amap

        self.au.write(self.retrive_n_address, 100)
        self.status_reg[number] = 0
        self.write_status()
        
        self.au.write(self.N_event_address, 0x00)
        self.au.write(self.init_n_address, number)
        self.init_buffer[:] = amap.flatten()

        self.au_event_fifo[number] = event
        packed_event = self.pack_event(event)
        self.in_fifo[:] = np.zeros(self.in_fifo.shape)
        self.in_fifo[0: packed_event.shape[0]] = packed_event
        self.au.write(self.init_fifo_depth_address, event.shape[0])

        
        self.au.write(0x00, 0x1)
        while not (self.au.read(0x0) & 0x2):
            pass
        self.read_status()


    def stream_in_events(self, events): 
        N = events.shape[0]
        self.allocate_event_buffer(N)
        packed_events = self.pack_event(events)
        self.event_buffer[:] = packed_events
        
        self.au.write(self.init_n_address, 100)
        self.au.write(self.N_event_address, N)
        self.au.write(self.retrive_n_address, 100)
        self.au.write(0x00, 0x01)
        begin = time.time()
        self.send.transfer(self.event_buffer)
        self.send.wait()
        
        while not (self.au.read(0x0) & 0x2):
            pass
        end = time.time()
        self.total_time += (end - begin)
        print("processing {} event using:{}".format(N, end-begin))
    
    def fifo_parser(self):
        out_fifo_idx = np.where(self.out_fifo!=0)
        out_events = self.out_fifo[out_fifo_idx]
        out_events = self.unpack_event(out_events)
        out_events = out_events[np.argsort(out_events[:, 2])]
        return out_events

    def read_amap(self, number):
        number = int(number)
        self.au.write(self.init_n_address, 100)
        self.au.write(self.N_event_address, 0)
        self.au.write(self.retrive_n_address, number)
        self.au.write(0x00, 0x01)
        while not (self.au.read(0x0) & 0x2):
            pass
    
    def dump_all_au(self):
        self.read_status()
        occupied_au = np.where(self.status_reg == 0)[0]
        for n in occupied_au:
            _,_ = self.dump_single_au(int(n))


    def dump_single_au(self, number):
        number = int(number)
        self.read_amap(number)
        self.amap[number, :] = self.output_buffer.reshape((self.Height, self.Width))
        self.au_event_fifo[number] = self.fifo_parser()
        return self.amap[number], self.au_event_fifo[number]

    def amapAddlocal(self, amap, x, y):
        b = self.dAdd
        x = int(x)
        y = int(y)
        idxx = np.arange(max(x - b, 0),
                         min(x + b + 1, self.Width))
        idxy = np.arange(max(y - b, 0),
                         min(y + b + 1, self.Height))
        idxxx, idxyy = np.meshgrid(idxx, idxy)
        amap[idxyy, idxxx] += 1
        return amap
        
    def write_all_au(self):
        live_au_list = np.where(self.status_reg == 0)[0]
        for i in live_au_list:
            auEvents = self.au_event_fifo[i]
            self.write_au(auEvents, i)


    def rebuild_amap_with_event(self, events): 
        amap = np.zeros([self.Height, self.Width], dtype=self.heatmap_precision)
        for event in events:
            self.amapAddlocal(amap, event[0], event[1])
        return amap
    
    def test_random(self, N, times):
        self.total_time = 0
        for t in range(times):
            x = np.random.randint(5, self.Width-5, size=N)
            y = np.random.randint(5, self.Height-5, size=N)
            t = np.ones(N)
            p = np.ones(N)
            events = np.vstack([x, y, t, p]).T  
            self.stream_in_events(events) 
        T = 1 / (self.total_time / N / times) /1000000
        print("Averaged Throughputs: {} Meps".format(T))
        return T
    
    def kill_au(self, number):
        number = int(number)
        self.au.write(self.retrive_n_address, 100)

        self.status_reg[number] = 1
        self.write_status()
        self.amap[number, :] = 0
        self.au_event_fifo[number][:] = 0
        self.auBox[number] = [0,0,0,0] 
        self.auNumber[number] = [0, 0] 

        self.au.write(self.N_event_address, 0x00)
        self.au.write(self.init_n_address, number)
        self.init_buffer[:] = 0

        self.in_fifo[:] = 0
        self.au.write(self.init_fifo_depth_address, 0)
        
        self.au.write(0x00, 0x1)
        while not (self.au.read(0x0) & 0x2):
            pass
        self.read_status()  

    def rebuild_all_amap(self):
        for i in range(self.au_number):
            self.amap[i] = self.rebuild_amap_with_event(self.au_event_fifo[i]) 
