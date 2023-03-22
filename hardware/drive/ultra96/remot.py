from au_functions import *
from au_hardware_fifo_only import Au_fifo
import numpy as np

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import squareform, directed_hausdorff
from itertools import combinations

from pynq import Overlay 
from pynq import allocate

class REMOT():
    def __init__(self, args):
        self.input = args.input
        self.bitfile = args.bitfile
        # self.Init(self.input)
        self.tkBoxes = []
        self.tkIDs = []

        self.auFifo = args.auFifo
        self.auNum = args.auNum
        self.dAdd = args.dAdd
        self.folder = args.outfolder
        self.name = args.name

        self.tFrame = args.tFrame
        self.lx = args.lx
        self.ly = args.ly

        self.bdspawn1 = args.bdspawn1
        self.bdspawn2 = args.bdspawn2
        self.bdkill = args.bdkill

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

        self.globalID = -1

        print("Bitfile = ", args.bitfile)
        self.AUs = Au_fifo(Height=self.ly, Width=self.lx, bitfile=args.bitfile, au_number=args.auNum, fifo_depth=args.auFifo)

    def update_box(self, ts):
        live_au_list = np.where(self.AUs.status_reg == 0)[0]
        for i in live_au_list:
            auEvents = self.AUs.au_event_fifo[i]
            idxFade = np.argwhere(auEvents[:, 2] < ts - self.tFrame).flatten()

            # idxFade = np.argwhere(auEvents[:, 2] < auEvents[-1, 2] - self.tFrame).flatten()
            if idxFade.size != 0:
                auEvents = np.delete(auEvents, idxFade, axis=0)
            self.AUs.au_event_fifo[i] = auEvents
            if auEvents.size > 0:
                self.AUs.auBox[i] = bbox(auEvents[:, 0], auEvents[:, 1])
            else:
                self.AUs.status_reg[i] = 1

    def Split(self):
        idxDel = []
        if (np.sum(self.AUs.status_reg) == 0): 
            print("All AU busy, can't split")
            return 
        
        for j in np.where(self.AUs.status_reg != 1)[0]:
            auEvents = self.AUs.au_event_fifo[j] 
            if auEvents.size == 0:
                continue
            if self.split == 'DBSCAN':
                idxGroup = DBSCAN(eps=self.epsDiv, min_samples=self.minptsDiv).fit_predict(
                    auEvents[:, :2])
                idxGroup[idxGroup < 0] = 0
            else:
                if auEvents.shape[0] <= 1:
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

            print(f"Identified {np.unique(idxGroup).size} clusters in AU {j}")

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
        
        idxGroup = clusterAu(np.array(self.AUs.auBox)[live_au_list], self.iomMer)
        # print("Merge idxGroup", idxGroup)
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
            if events.size > 0:
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
            if self.AUs.au_event_fifo[idx].size == 0:
                idxDel.append(idx)
                continue
            flag1 = ts - np.max(self.AUs.au_event_fifo[idx][:, 2]) > self.tDel
            flag2 = bbArea(self.AUs.auBox[idx]) < self.areaDel
            flag3 = (self.AUs.auBox[idx][1] + self.AUs.auBox[idx][3]) / 2 < self.bdkill
            if flag1 or flag2 or flag3:
                idxDel.append(idx)
                print(idx, "is killed due to ", "timeout: ", flag1, "size: ", flag2, "bdkill: ", flag3)
                # print("ts: ", ts)
                # print("np.max(self.AUs.au_event_fifo[idx][:, 2]): ", np.max(self.AUs.au_event_fifo[idx][:, 2]))
                                 
                              
        if len(idxDel) > 0:
            # print(ts)
            # print("killing", idxDel)
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

    def Process(self, events, dump_au):
        ts = events[-1, 2].astype(np.uint32)
        self.AUs.stream_in_events(events)

        if dump_au:
            self.AUs.dump_all_au()
            self.update_box(ts)
            self.Merge()
            self.Kill(ts)
            self.Split()
            self.UpdateID(ts)
        # live AU, (AU ID, timestamp), AU fifo

        live_au_list = np.where(self.AUs.status_reg==0)[0]

        return (live_au_list, self.AUs.auNumber, self.AUs.au_event_fifo)