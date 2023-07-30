import tkinter as tk
from tkinter import *
import customtkinter

import cv2
import PIL.Image, PIL.ImageTk
import pyrealsense2 as rs
import numpy as np

# System settings
customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.title("Realsense")
app.bind("<Escape>", lambda e: app.quit())


# UI elements
label = customtkinter.CTkLabel(app, text="D435 Depth Stream")
label.pack()

container = customtkinter.CTkFrame(app)
container.pack()

pipeline = rs.pipeline()
config = rs.config()
config.enable_device("233722072412")
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
advnc_mode = rs.rs400_advanced_mode(device)
current_std_depth_table = advnc_mode.get_depth_table()
current_std_depth_table.depthClampMin = 0
current_std_depth_table.depthClampMax = 800
advnc_mode.set_depth_table(current_std_depth_table)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

align = rs.align(rs.stream.depth)

# Camera LEFT
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device("032622070359")
pipeline_wrapper_2 = rs.pipeline_wrapper(pipeline_2)
pipeline_profile_2 = config_2.resolve(pipeline_wrapper_2)
device_2 = pipeline_profile_2.get_device()
advnc_mode_2 = rs.rs400_advanced_mode(device_2)
current_std_depth_table_2 = advnc_mode_2.get_depth_table()
current_std_depth_table_2.depthClampMin = 0
current_std_depth_table_2.depthClampMax = 800
advnc_mode_2.set_depth_table(current_std_depth_table_2)
config_2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.depth)

cfg_2 = pipeline_2.start(config_2)
profile_2 = cfg_2.get_stream(rs.stream.depth)


canvas_depth = customtkinter.CTkCanvas(container, width=1280, height=720, bg="white")
canvas_depth.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
canvas_RGB = customtkinter.CTkCanvas(container, width=1280, height=720, bg="white")
canvas_RGB.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
canvas_depth_2 = customtkinter.CTkCanvas(container, width=1280, height=720, bg="white")
canvas_depth_2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
canvas_RGB_2 = customtkinter.CTkCanvas(container, width=1280, height=720, bg="white")
canvas_RGB_2.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
frame_RGB = None
frame_RGB_2 = None
frame_depth = None
frame_depth_2 = None
photo_RGB = None
photo_RGB_2 = None
photo_depth = None
photo_depth_2 = None

colorizer = rs.colorizer()


def open_camera():
    global canvas_depth, canvas_depth_2, canvas_RGB, canvas_RGB_2, photo_RGB, photo_RGB_2, photo_depth, photo_depth_2, frame_RGB, frame_RGB_2, frame_depth, frame_depth_2

    frames = pipeline.wait_for_frames()
    frames_2 = pipeline_2.wait_for_frames()

    aligned_frames = align.process(frames)
    aligned_frames_2 = align.process(frames_2)

    depth_frame = aligned_frames.get_depth_frame()
    depth_frame_2 = aligned_frames_2.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_frame_2 = aligned_frames_2.get_color_frame()

    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    depth_image_2 = np.asanyarray(colorizer.colorize(depth_frame_2).get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image_2 = np.asanyarray(color_frame_2.get_data())

    frame_RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    frame_RGB_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2RGB)
    frame_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    frame_depth_2 = cv2.cvtColor(depth_image_2, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("sao.jpg",frame)

    photo_RGB = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_RGB))
    photo_RGB_2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_RGB_2))
    photo_depth = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_depth))
    photo_depth_2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_depth_2))

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    canvas_RGB.create_image(0, 0, image=photo_RGB, anchor=NW)
    canvas_RGB_2.create_image(0, 0, image=photo_RGB_2, anchor=NW)
    canvas_depth.create_image(0, 0, image=photo_depth, anchor=NW)
    canvas_depth_2.create_image(0, 0, image=photo_depth_2, anchor=NW)
    app.after(10, open_camera)


def save_frame():
    print("prova")
    if frame_RGB is not None:
        cv2.imwrite("sao.jpg", frame_RGB)


app.bind("<s>", lambda _: save_frame())
app.bind("<S>", lambda _: save_frame())

open_camera()
app.mainloop()
