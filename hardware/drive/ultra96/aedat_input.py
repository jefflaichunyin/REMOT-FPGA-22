import yaml
from au_controller import Controller
from au_functions import ARGS
import threading
import argparse
import datetime
import time
import cv2
import numpy as np
import cProfile

from threading import Event
from pycaer.aedat import Aedat
import sys

aedat_file = sys.argv[1]
aedat = Aedat(aedat_file)

config_dir = "./config/shape_6dof_fifo.yml"

with open(config_dir, "r") as file:
    config = yaml.safe_load(file)

config = ARGS(config)
controller = Controller(config)

### IMPORTANT ###
# Run sudo systemctl start dv-runtime.service
# And Check if it is running with DV-GUI
### IMPORTANT ###

#FPS is hard coded to be 25 in au_controller code. See AU_Init tFrame setting
fps = 30

# Redefine the Camera Parameters Here
pixel_length_x = 346
pixel_length_y = 260


# Set the Event Camera Resolution
controller.lx = pixel_length_x
controller.ly = pixel_length_y

# nFrame number is not used in au_controller. Large number would only occupy memory
controller.nFrame = 1 # should be the number of frame here
controller.frame0 = np.zeros((controller.ly, controller.lx, 3), 'uint8') + 255
controller.iFrame = controller.frame0.copy()
#"frames" are not used in au_controller.py
controller.frame_count = 0
controller.tFrame = int(1/fps * 1000000) 

#Check if tDel looks okay
print("tDel ", controller.tDel)

#Check if FPGA bitstream is successfully flashed
controller.AUs.test_random(32768, 100)

i = 0
dump_interval = 5

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('frame', 346*2, 260*2)
while True:
    # events are formated in the following way: [x, y, self.t, p]
    frame = np.full((260,346,3), 0, 'uint8')
    events = aedat.read(frame)
    #cProfile.run('events = davis.read()', 'read.prof')
    if len(events) > 0:
        # controller.StreamEvents(events)
        dump_au = (i==dump_interval)
        i = 0 if dump_au else (i+1)
        controller.Process(events, frame, dump_au)
        pass
        # cProfile.run("controller.Process(events)", 'main.prof')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break