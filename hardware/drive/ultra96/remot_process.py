import numpy as np
from au_functions import *

from sklearn.cluster import DBSCAN, AgglomerativeClustering


def write_au(status, fifo, event, number, args):
    event = event[-args.auFifo:]
    number = int(number)
    status[number] = 0
    fifo[number] = event

def kill_au(status, fifo, box, number, au_id):
    box[au_id] = [0,0,0,0] 
    number[au_id] = [0, 0] 
    status[au_id] = 1
    fifo[au_id][:] = 0 

def update_box(ts, status, fifo, box, args):
    live_au_list = np.where(status == 0)[0]
    for i in live_au_list:
        auEvents = fifo[i]
        idxFade = np.argwhere(auEvents[:, 2] < ts - args.tFrame).flatten()

        # idxFade = np.argwhere(auEvents[:, 2] < auEvents[-1, 2] - self.tFrame).flatten()
        if idxFade.size != 0:
            auEvents = np.delete(auEvents, idxFade, axis=0)
        fifo[i] = auEvents
        if auEvents.size > 0:
            box[i] = bbox(auEvents[:, 0], auEvents[:, 1])
        else:
            status[i] = 1

def Split(status, fifo, box, number, args):
    idxDel = []
    if (np.sum(status) == 0): 
        print("All AU busy, can't split")
        return 
    
    for j in np.where(status!= 1)[0]:
        auEvents = fifo[j] 
        if auEvents.size == 0:
            continue

        idxGroup = DBSCAN(eps=args.epsDiv, min_samples=args.minptsDiv).fit_predict(
            auEvents[:, :2])
        idxGroup[idxGroup < 0] = 0


        if max(idxGroup) <= 0: 
            continue
        else:
            idxDel.append(j)

        # print(f"Identified {np.unique(idxGroup).size} clusters in AU {j}")

        # find cluster with most events
        idxTk = np.argmax([sum(idxGroup == idx)
                            for idx in np.unique(idxGroup)])

        for k in range(max(idxGroup) + 1): # go through each cluster
            next_empty = np.where(status== 1)[0]
            if next_empty.shape[0] ==0:
                return
            else:
                next_empty = next_empty[0]
                
                idxEvents = np.argwhere(idxGroup == k).flatten()
                event_collect = auEvents[idxEvents]
                box[next_empty] = bbox(event_collect[:, 0], event_collect[:, 1]) 
                
                write_au(status, fifo, event_collect, next_empty, args)
                if k == idxTk:
                    number[next_empty] = number[j] 
                else:
                    number[next_empty] = [0, min(event_collect[:, 2])]

    if len(idxDel) > 0:
        for idx in (idxDel):
            kill_au(status, fifo, box, number, idx)
            
def Merge(status, fifo, box, number, args):
    live_au_list = np.where(status == 0)[0]
    if live_au_list.shape[0] <= 1: # if state reg only has one 0
        return
    live_au_fifo = [fifo[i] for i in live_au_list]
    idxGroup = clusterAu_hausdroff(live_au_fifo, args.dsMer)
    # idxGroup = clusterAu(np.array(box)[live_au_list], args.iomMer)
    # print("Merge idxGroup", idxGroup)
    for j in range(max(idxGroup) + 1):
        idxAU = np.argwhere(idxGroup == j).flatten()
        idxAU = live_au_list[idxAU]
        idxDel = idxAU
        if idxAU.size < 2:
            continue
        
        write_au_idx = np.min(idxAU)
        # print("merging{} to {}".format(idxAU, write_au_idx))
        events = np.concatenate(
            [fifo[idx] for idx in idxAU], axis=0) 
        events = np.unique(events, axis=0)
        # events = events[np.argsort(events[:, 2])]
    
        if any([number[idx][0] > 0 for idx in idxAU]):
            idxAU = idxAU[[number[idx][0] > 0 for idx in idxAU]]
        idxNum = idxAU[np.argmin([number[idx][0] for idx in idxAU])]
        
        write_au(status, fifo, events, write_au_idx, args)
        if events.size > 0:
            box[write_au_idx] = bbox(events[:, 0], events[:, 1])
        number[write_au_idx] = number[idxNum]
        
        for idx in idxDel:
            # print("killing", idxDel)

            if idx == write_au_idx:
                continue
            kill_au(status, fifo, box, number, idx)

def Kill(ts, status, fifo, box, number, args):
    live_au_list = np.where(status==0)[0]
    idxDel = []
    for idx in live_au_list:
        if fifo[idx].size == 0:
            idxDel.append(idx)
            continue
        ts = int(ts)
        max_ts = int(np.max(fifo[idx][:, 2]))
        tDel = args.tDel * args.tFrame
        flag1 = ts - max_ts > tDel
        flag2 = bbArea(box[idx]) < args.areaDel
        flag3 = (box[idx][1] + box[idx][3]) / 2 < args.bdkill
        if flag1 or flag2 or flag3:
            idxDel.append(idx)
            # print(idx, "is killed due to ", "timeout: ", flag1, "size: ", flag2, bbArea(box[idx]), "bdkill: ", flag3)
            # print("ts: ", ts)
            # print("np.max(self.AUs.au_event_fifo[idx][:, 2]): ", np.max(fifo[idx][:, 2]))
                                
                            
    if len(idxDel) > 0:
        # print(ts)
        # print("killing", idxDel)
        for idx in idxDel:
            kill_au(status, fifo, box, number, idx)

def UpdateID(ts, status, fifo, number, box, global_id, args):
    live_au_list = np.where(status==0)[0]
    for idx in live_au_list:
        # print(f'AU{idx} ID{number[idx][0]} life {int(ts - number[idx][1])} area {bbArea(box[idx])} event {fifo[idx].shape[0]}')
        if not number[idx][0]:
            life_enough = int(ts - number[idx][1]) > args.tLive * args.tFrame
            area_enough = bbArea(box[idx]) > args.areaLive
            event_enough = fifo[idx].shape[0] > args.numLive
            if life_enough and area_enough and event_enough:
                global_id += 1
                number[idx][0] = global_id
                # print(f'Assigned {global_id}')
            # else:
                # print(f'ID not assigned due to life: {life_enough} area: {area_enough} event: {event_enough}')
    return global_id

def REMOT_Update(ts, status, fifo, number, global_id, args):
    au_box = [[0,0,0,0] for i in range(args.auNum)]
    update_box(ts, status, fifo, au_box, args)
    # Kill(ts, status, fifo, au_box, number, args)
    return UpdateID(ts, status, fifo, number, au_box, global_id, args)

def REMOT_Process(ts, status, fifo, number, global_id, args):
    # print("REMOT_Process", status)
    au_box = [[0,0,0,0] for i in range(args.auNum)]
    update_box(ts, status, fifo, au_box, args)
    # Merge(status, fifo, au_box, number, args)
    # Kill(ts, status, fifo, au_box, number, args)
    Split(status, fifo, au_box, number, args)
    Merge(status, fifo, au_box, number, args)
    Kill(ts, status, fifo, au_box, number, args)

    update_box(ts, status, fifo, au_box, args)
    global_id = UpdateID(ts, status, fifo, number, au_box, global_id, args)

    return (status, fifo, number, global_id)
    # return (self.AUs.status_reg, self.AUs.au_event_fifo, self.AUs.au_number)