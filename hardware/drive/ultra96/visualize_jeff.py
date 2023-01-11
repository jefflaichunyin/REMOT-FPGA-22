import yaml
from au_controller import Controller
from au_functions import ARGS
import threading
import argparse
import datetime
import time
import cv2
import numpy as np
from pycaer import davis
from threading import Event

davis.open()

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
fps = 25

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

while True:
    # events are formated in the following way: [x, y, self.t, p]
    events = davis.read()
    if len(events) > 0:
        controller.Process(events)
    cv2.imshow('frame', controller.iFrame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break