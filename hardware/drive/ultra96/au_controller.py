from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy import io
from scipy.spatial.distance import squareform, directed_hausdorff
from itertools import combinations
import numpy as np
from pynq import Overlay 
from pynq import allocate
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import os
import csv
import yaml
from au_functions import *
from au_hardware_hash import Au_hash
from au_hardware_full import Au_full
from au_hardware_fifo_only import Au_fifo
import threading
import time

class Controller():
    def __init__(self, args):
        self.input = args.input
        self.bitfile = args.bitfile
        self.Init(self.input)
        self.tkBoxes = []
        self.tkIDs = []

        self.auFifo = args.auFifo
        self.auNum = args.auNum
        self.dAdd = args.dAdd
        self.folder = args.outfolder
        self.name = args.name

        self.tDel = args.tDel * self.tFrame
        self.areaDel = args.areaDel
        self.tLive = args.tLive * self.tFrame
        self.areaLive = args.areaLive
        self.numLive = args.numLive

        self.t = self.tFrame

        self.split = args.split
        self.epsDiv = args.epsDiv
        self.minptsDiv = 1

        self.merge = args.merge
        self.iomMer = args.iomMer
        self.dsMer = args.dsMer
        self.minptsMer = 1
        self.epsMer = 1

        if "inbound" in self.input:
            self.bdspawn1 = 40
            self.bdspawn2 = 40
        else:
            self.bdspawn1 = -1
            self.bdspawn2 = -1

        if "outbound" in self.input:
            self.bdkill = 20
        else:
            self.bdkill = -1

        self.globalID = -1
        print("Bitfile = ", args.bitfile)
        if 'hash' in args.bitfile:
            self.AUs = Au_hash(Height=self.ly, Width=self.lx, bitfile=args.bitfile, au_number=args.auNum, fifo_depth=args.auFifo, dAdd=self.dAdd)
        elif 'full' in args.bitfile:
            self.AUs = Au_full(Height=260, Width=342, bitfile=args.bitfile, au_number=args.auNum, fifo_depth=args.auFifo, dAdd=self.dAdd)
        elif 'fifo' in args.bitfile:
            self.AUs = Au_fifo(Height=self.ly, Width=self.lx, bitfile=args.bitfile, au_number=args.auNum, fifo_depth=args.auFifo)
        else:
            raise ValueError("Not matching bitfile overlay")

        self.frame_count = 0
        x = self.events[:, 0]
        y = self.events[:, 1]
        p = self.events[:, 2]
        self.t = self.events[:, 3]
        self.total_time = self.t[-1]
        self.events = np.vstack([x, y, self.t, p]).T

        self.frame_lock = threading.Lock()
            
            
    def Init(self, file):
        if "bound" in self.input:
            self.tFrame = 40000
        else:
            self.tFrame = 44065

        self.cmap = io.loadmat('cmap.mat')['cmap'] * 255
        self.events = io.loadmat(file)['events']
        self.lx, self.ly = self.events[:, :2].max(0) + 1
        self.pFrame = -1
        self.nFrame = self.events[:, 4][-1] + 1
        self.frame0 = np.zeros((self.ly, self.lx, 3), 'uint8')
        self.iFrame = self.frame0.copy()

    def update_box(self, ts):
        live_au_list = np.where(self.AUs.status_reg == 0)[0]
        for i in live_au_list:
            auEvents = self.AUs.au_event_fifo[i]
            idxFade = np.argwhere(auEvents[:, 2] < ts - self.tFrame).flatten()

            idxFade = np.argwhere(auEvents[:, 2] < auEvents[-1, 2] - self.tFrame).flatten()
            if idxFade.size != 0:
                auEvents = np.delete(auEvents, idxFade, axis=0)
            self.AUs.au_event_fifo[i] = auEvents
            self.AUs.auBox[i] = bbox(auEvents[:, 0], auEvents[:, 1])
                
        
    def Split(self):
        idxDel = []
        if (np.sum(self.AUs.status_reg) == 0): 
            return 
        
        for j in np.where(self.AUs.status_reg != 1)[0]:
            auEvents = self.AUs.au_event_fifo[j] 
            if self.split == 'DBSCAN':
                idxGroup = DBSCAN(eps=self.epsDiv, min_samples=self.minptsDiv).fit_predict(
                    auEvents[:, :2])
                idxGroup[idxGroup < 0] = 0
            else:
                if au.auEvents.shape[0] <= 1:
                    continue
                clustering = AgglomerativeClustering(
                    linkage='average', affinity='euclidean').fit(auEvents[:, :2])
                idxGroup = clustering.labels_
                idxGroup[idxGroup < 0] = 0
                if max(directed_hausdorff(auEvents[idxGroup == 0, :2], auEvents[idxGroup == 1, :2])[0],
                       directed_hausdorff(auEvents[idxGroup == 1, :2], auEvents[idxGroup == 0, :2])[0]) < self.dsMer:
                    continue

            if max(idxGroup) <= 0: 
                continue
            else:
                idxDel.append(j)

            idxTk = np.argmax([sum(idxGroup == idx)
                              for idx in np.unique(idxGroup)])

            for k in range(max(idxGroup) + 1):
                next_empty = np.where(self.AUs.status_reg == 1)[0]
                if next_empty.shape[0] ==0:
                    return
                else:
                    next_empty = next_empty[0]
                    
                    idxEvents = np.argwhere(idxGroup == k).flatten()
                    event_collect = auEvents[idxEvents]
                    self.AUs.auBox[next_empty] = bbox(event_collect[:, 0], event_collect[:, 1]) 
                    
                    self.AUs.write_au(event=event_collect, number=next_empty)
                    if k == idxTk:
                        self.AUs.auNumber[next_empty] = self.AUs.auNumber[j] 
                    else:
                        self.AUs.auNumber[next_empty] = [0, min(event_collect[:, 2])]

        if len(idxDel) > 0:
            for idx in (idxDel):
                self.AUs.kill_au(idx)
                
    def Merge(self):
        live_au_list = np.where(self.AUs.status_reg == 0)[0]
        if live_au_list.shape[0] <= 1: ## if state reg only has one 0
            return
        
        idxnk = list(combinations(live_au_list, 2)) 
        idxGroup = clusterAu(np.array(self.AUs.auBox)[live_au_list], self.iomMer)

        for j in range(max(idxGroup) + 1):
            idxAU = np.argwhere(idxGroup == j).flatten()
            idxAU = live_au_list[idxAU]
            idxDel = idxAU
            if idxAU.size < 2:
                continue
            
            write_au_idx = np.min(idxAU)
            print("merging{} to {}".format(idxAU, write_au_idx) )
            events = np.concatenate(
                [self.AUs.au_event_fifo[idx] for idx in idxAU], axis=0) 
            events = np.unique(events, axis=0)
            events = events[np.argsort(events[:, 2])]
        
            if any([self.AUs.auNumber[idx][0] > 0 for idx in idxAU]):
                idxAU = idxAU[[self.AUs.auNumber[idx][0] > 0 for idx in idxAU]]
            idxNum = idxAU[np.argmin([self.AUs.auNumber[idx][0] for idx in idxAU])]
            
            self.AUs.write_au(event=events, number=write_au_idx)
            self.AUs.auBox[write_au_idx] = bbox(events[:, 0], events[:, 1])
            self.AUs.auNumber[write_au_idx] = self.AUs.auNumber[idxNum]
            
            for idx in idxDel:
                if idx == write_au_idx:
                    continue
                self.AUs.kill_au(idx)

    def Kill(self, ts):
        live_au_list = np.where(self.AUs.status_reg==0)[0]
        idxDel = []
        
        for idx in live_au_list:
            flag1 = ts - np.max(self.AUs.au_event_fifo[idx][:, 2]) > self.tDel
            flag2 = bbArea(self.AUs.auBox[idx]) < self.areaDel
            flag3 = (self.AUs.auBox[idx][1] + self.AUs.auBox[idx][3]) / 2 < self.bdkill
            if flag1 or flag2 or flag3:
                idxDel.append(idx)
                                 
                              
        if len(idxDel) > 0:
            print("killing", idxDel)
            for idx in idxDel:
                self.AUs.kill_au(idx)  

    def UpdateID(self, ts):
        if 'fifo' in self.bitfile:
            self.AUs.write_all_au()
            
        live_au_list = np.where(self.AUs.status_reg==0)[0]
        for idx in live_au_list:
            if not self.AUs.auNumber[idx][0] and \
            ts - self.AUs.auNumber[idx][1] > self.tLive and \
            bbArea(self.AUs.auBox[idx]) > self.areaLive and \
            self.AUs.au_event_fifo[idx].shape[0] > self.numLive and \
            self.AUs.auBox[idx][2] / 2 > self.bdspawn1 and \
            self.AUs.auBox[idx][3] / 2 > self.bdspawn2:
                self.globalID += 1
                self.AUs.auNumber[idx][0] = self.globalID

    def Animation(self, events, ts):
        self.iFrame[:] = 0
        y = events[:, 1]
        x = events[:, 0]
        self.iFrame[y, x] = [255, 255, 255]
        
        idxVis = []
        boxes = []
        IDs = []

        live_au_list = np.where(self.AUs.status_reg == 0)[0]
        cmap_idx = np.array([self.AUs.auNumber[i][0] % 7 for i in live_au_list], 'int32')
        auColors = np.zeros([self.auNum, 3])
        auColors[live_au_list] = self.cmap[cmap_idx]

        for j in live_au_list:
            if self.AUs.auNumber[j][0] > 0:
                idxEvt = self.AUs.au_event_fifo[j][:, 2] >= ts - self.tFrame
                if any(idxEvt):
                    idxVis.append(j)
                    one_frame_events = self.AUs.au_event_fifo[j][idxEvt, :]
                    boxes.append(
                        bbox(one_frame_events[:, 0], one_frame_events[:, 1]))
                    IDs.append(self.AUs.auNumber[j][0])

        self.tkBoxes.append(boxes)
        self.tkIDs.append(IDs)
        print("tkid", IDs)

        if len(idxVis) > 0:
            for j, k in enumerate(idxVis):
                self.iFrame = cv2.rectangle(
                    self.iFrame, (boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), auColors[k].tolist(), 1)

        if len(idxVis) > 0:
            for j, k in enumerate(idxVis):
                self.iFrame = cv2.putText(self.iFrame, '{}'.format(
                    IDs[j]), (boxes[j][0], boxes[j][1]), cv2.FONT_HERSHEY_PLAIN, 1., auColors[k].tolist(), 1)

    def Process(self, events):
        ts = events[-1, 2]
        self.AUs.stream_in_events(events)
        self.AUs.dump_all_au()
        self.update_box(ts)
        
        self.Split()
        self.Merge()
        self.Kill(ts)
        self.UpdateID(ts)
        with self.frame_lock:
            self.Animation(events, ts)

    def get_frame(self):
        with self.frame_lock:
            return self.iFrame

    def Process_all(self):
        number_of_frame = self.total_time // (1 * self.tFrame)
        for i in range(number_of_frame):
            print("Processing {}---------------------------------------".format(self.frame_count))
            idxs = np.where((self.t > self.frame_count * self.tFrame) & (self.t <= (self.frame_count + 1) * 1 * self.tFrame))[0]
            self.Process(self.events[idxs])
            self.frame_count += 1
            time.sleep(self.tFrame/1e6)
        print("Done")