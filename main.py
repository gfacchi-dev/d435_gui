import tkinter as tk
from tkinter import *

import cv2
import PIL.Image, PIL.ImageTk
import pyrealsense2 as rs
import numpy as np

window = Tk()
window.title("Realsense")
window.bind("<Escape>", lambda e: window.quit())

pipeline = rs.pipeline()
config = rs.config()

freq = 30
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg = pipeline.start(config)
dev = cfg.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
align_to = rs.stream.color
align = rs.align(align_to)


from threading import Thread

canvas = Canvas(window, width=1280, height=720, bg="white")
canvas.pack()
photo = None

colorizer = rs.colorizer()


def open_camera():
    global canvas, photo

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    color_image = np.asanyarray(color_frame.get_data())
    # frame=cv2.resize(color_image,dsize=None,fx=0.5,fy=0.5)
    # frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("sao.jpg",frame)

    # photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    canvas.create_image(0, 0, image=photo, anchor=NW)
    window.after(10, open_camera)


open_camera()
window.mainloop()
