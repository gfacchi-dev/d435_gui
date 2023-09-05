from tkinter import *
import customtkinter
from utils import (
    FrameQueue,
    mean_variance_and_inliers_along_third_dimension_ignore_zeros,
    get_maps,
    save_pcl,
)

import cv2
import PIL.Image, PIL.ImageTk
import pyrealsense2 as rs
import numpy as np
import pickle
import open3d as o3d
import math
import itertools
import os, datetime

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
intr_right = profile.as_video_stream_profile().get_intrinsics()

cfg_2 = pipeline_2.start(config_2)
profile_2 = cfg_2.get_stream(rs.stream.depth)
intr_left = profile_2.as_video_stream_profile().get_intrinsics()

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
color_image = None
color_image_2 = None

colorizer = rs.colorizer()

# write a function that queues 90 frames and if an element added exceeds dimensions autoremoves last element


depth_queue = FrameQueue(max_frames=90, frame_shape=(720, 1280))
depth_queue_2 = FrameQueue(max_frames=90, frame_shape=(720, 1280))
rgb_queue = FrameQueue(max_frames=1, frame_shape=(720, 1280, 3))
rgb_queue_2 = FrameQueue(max_frames=1, frame_shape=(720, 1280, 3))


def process_frames(frames, pipeline, colorizer, canvas_RGB, canvas_depth, depth_queue):
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    color_image = np.asanyarray(color_frame.get_data())

    frame_RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    frame_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)

    photo_RGB = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_RGB))
    photo_depth = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_depth))

    canvas_RGB.create_image(0, 0, image=photo_RGB, anchor=NW)
    canvas_depth.create_image(0, 0, image=photo_depth, anchor=NW)

    depth_queue.add_frame(np.asanyarray(depth_frame.get_data()))
    print(len(depth_queue))


def open_camera():
    global color_image, color_image_2, canvas_depth, canvas_depth_2, canvas_RGB, canvas_RGB_2, photo_RGB, photo_RGB_2, photo_depth, photo_depth_2, frame_RGB, frame_RGB_2, frame_depth, frame_depth_2, depth_queue, depth_queue_2

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    frames_2 = pipeline_2.wait_for_frames()
    frames_2 = align.process(frames_2)

    depth_frame = frames.get_depth_frame()
    depth_frame_2 = frames_2.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_frame_2 = frames_2.get_color_frame()

    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    depth_image_2 = np.asanyarray(colorizer.colorize(depth_frame_2).get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image_2 = np.asanyarray(color_frame_2.get_data())

    frame_RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    frame_RGB_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2RGB)
    frame_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    frame_depth_2 = cv2.cvtColor(depth_image_2, cv2.COLOR_BGR2RGB)

    photo_RGB = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_RGB))
    photo_RGB_2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_RGB_2))
    photo_depth = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_depth))
    photo_depth_2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_depth_2))

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    canvas_RGB.create_image(0, 0, image=photo_RGB, anchor=NW)
    canvas_RGB_2.create_image(0, 0, image=photo_RGB_2, anchor=NW)
    canvas_depth.create_image(0, 0, image=photo_depth, anchor=NW)
    canvas_depth_2.create_image(0, 0, image=photo_depth_2, anchor=NW)

    depth_queue.add_frame(np.asanyarray(depth_frame.get_data()))
    depth_queue_2.add_frame(np.asanyarray(depth_frame_2.get_data()))
    rgb_queue.add_frame(np.asanyarray(color_image))
    rgb_queue_2.add_frame(np.asanyarray(color_image))
    app.after(10, open_camera)


def calibrate():
    (
        mean_left,
        variance_left,
        inliers_left,
    ) = mean_variance_and_inliers_along_third_dimension_ignore_zeros(
        depth_queue.get_frames_as_tensor()
    )
    mean_left = np.asarray(mean_left.numpy(), dtype=np.uint16)
    variance_left = np.asarray(variance_left.numpy(), dtype=np.uint16)
    inliers_left = np.asarray(inliers_left.numpy(), dtype=np.uint16)
    (
        mean_right,
        variance_right,
        inliers_right,
    ) = mean_variance_and_inliers_along_third_dimension_ignore_zeros(
        depth_queue_2.get_frames_as_tensor()
    )
    mean_right = np.asarray(mean_right.numpy(), dtype=np.uint16)
    variance_right = np.asarray(variance_right.numpy(), dtype=np.uint16)
    inliers_right = np.asarray(inliers_right.numpy(), dtype=np.uint16)

    variance_image_l, zero_variance_image_l, threshold_l, filtered_means_l = get_maps(
        variance_left,
        mean_left,
        1,
    )
    variance_image_r, zero_variance_image_r, threshold_r, filtered_means_r = get_maps(
        variance_right,
        mean_right,
        1,
    )

    from scipy.interpolate import interp1d

    indexes = np.argwhere(variance_image_l == 255)
    selected_m_l = np.copy(mean_left)
    selected_m_l[indexes[:, 0], indexes[:, 1]] = 0
    cv2.imwrite("temp/leftDepth.png", np.uint16(selected_m_l))
    cv2.imwrite("temp/leftColor.jpg", color_image)

    indexes = np.argwhere(variance_image_r == 255)
    selected_m_r = np.copy(mean_right)
    selected_m_r[indexes[:, 0], indexes[:, 1]] = 0
    cv2.imwrite("temp/rightDepth.png", np.uint16(selected_m_r))
    cv2.imwrite("temp/rightColor.jpg", color_image_2)

    depth_raw_left = o3d.io.read_image("temp/leftDepth.png")
    color_raw_left = o3d.io.read_image("temp/leftColor.jpg")
    depth_raw_right = o3d.io.read_image("temp/rightDepth.png")
    color_raw_right = o3d.io.read_image("temp/rightColor.jpg")
    rgbd_image_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_left, depth_raw_left
    )
    rgbd_image_right = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_right, depth_raw_right
    )

    camera_intrinsic_left = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_left.width,
            intr_left.height,
            intr_left.fx,
            intr_left.fy,
            intr_left.ppx,
            intr_left.ppy,
        )
    )
    camera_intrinsic_right = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_right.width,
            intr_right.height,
            intr_right.fx,
            intr_right.fy,
            intr_right.ppx,
            intr_right.ppy,
        )
    )

    pcd_left = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_left, camera_intrinsic_left
    )
    pcd_right = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_right, camera_intrinsic_right
    )

    bounds = [
        [-math.inf, math.inf],
        [-math.inf, math.inf],
        [0.2, 0.7],
    ]  # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points)
    )  # create bounding box object

    # Crop the point cloud using the bounding box:
    pcd_left = pcd_left.crop(bounding_box)
    pcd_right = pcd_right.crop(bounding_box)

    # Rotazione di 180° intorno all'asse X (ribaltare la point cloud poiché la pinhole camera ribalta la visuale)
    angolo = np.pi
    trans_x = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angolo), -np.sin(angolo), 0.0],
            [0.0, np.sin(angolo), np.cos(angolo), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    pcd_left.transform(trans_x)
    pcd_right.transform(trans_x)

    # Primario -> Camera di destra
    # Secondario -> Camera di sinistra

    # Rotazione di -30° intorno all'asse Y
    angolo = np.pi / 6
    trans_y = np.asarray(
        [
            [np.cos(angolo), 0.0, np.sin(angolo), -0.34],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angolo), 0.0, np.cos(angolo), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    source = pcd_left
    target = pcd_right
    # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=400))
    # target.orient_normals_consistent_tangent_plane(50)
    threshold = 0.01
    trans_init = np.asarray(
        [
            [np.cos(angolo), 0.0, -np.sin(angolo), -0.31],
            [0.0, 1.0, 0.0, 0.0],
            [np.sin(angolo), 0.0, np.cos(angolo), -0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    print(evaluation)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=1000000, relative_rmse=0.001
        ),
    )
    print(reg_p2p)
    calibrated_matrix = reg_p2p.transformation

    pickle.dump(calibrated_matrix, open("temp/cal_mat.mat", "wb"))


def acquire(mesh=False):
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir("./acquisizioni/" + datestring)
    matrice_calibrazione = pickle.load(open("temp/cal_mat.mat", "rb"))

    depth_frame_l = depth_queue_2.get_last_frame()
    color_frame_l = rgb_queue_2.get_last_frame()
    print(depth_frame_l.dtype)
    depth_frame_r = depth_queue.get_last_frame()
    color_frame_r = rgb_queue.get_last_frame()

    cv2.imwrite(f"./acquisizioni/{datestring}/d_l.png", depth_frame_l)
    cv2.imwrite(f"./acquisizioni/{datestring}/rgb_l.png", color_frame_l)

    cv2.imwrite(f"./acquisizioni/{datestring}/d_r.png", depth_frame_r)
    cv2.imwrite(f"./acquisizioni/{datestring}/rgb_r.png", color_frame_r)

    depth_raw_left = o3d.io.read_image(f"./acquisizioni/{datestring}/d_l.png")
    color_raw_left = o3d.io.read_image(f"./acquisizioni/{datestring}/rgb_l.png")
    depth_raw_right = o3d.io.read_image(f"./acquisizioni/{datestring}/d_r.png")
    color_raw_right = o3d.io.read_image(f"./acquisizioni/{datestring}/rgb_r.png")

    rgbd_image_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_left, depth_raw_left, convert_rgb_to_intensity=False
    )
    rgbd_image_right = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_right, depth_raw_right, convert_rgb_to_intensity=False
    )

    camera_intrinsic_left = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_left.width,
            intr_left.height,
            intr_left.fx,
            intr_left.fy,
            intr_left.ppx,
            intr_left.ppy,
        )
    )
    camera_intrinsic_right = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_right.width,
            intr_right.height,
            intr_right.fx,
            intr_right.fy,
            intr_right.ppx,
            intr_right.ppy,
        )
    )

    pcd_left = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_left, camera_intrinsic_left
    )
    pcd_right = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_right, camera_intrinsic_right
    )

    bounds = [
        [-math.inf, math.inf],
        [-math.inf, math.inf],
        [0.1, 0.7],
    ]  # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points)
    )  # create bounding box object

    # Crop the point cloud using the bounding box:
    pcd_left = pcd_left.crop(bounding_box)
    pcd_right = pcd_right.crop(bounding_box)

    # Rotazione di 180° intorno all'asse X (ribaltare la point cloud poiché la pinhole camera ribalta la visuale)
    angolo = np.pi
    trans_x = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angolo), -np.sin(angolo), 0.0],
            [0.0, np.sin(angolo), np.cos(angolo), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    pcd_left.transform(trans_x)
    pcd_right.transform(trans_x)

    pcd_left.transform(matrice_calibrazione)
    save_pcl(pcd_left, pcd_right, datestring)

    if mesh:
        pcd = o3d.io.read_point_cloud(f"acquisizioni/{datestring}/pcl.pcd")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=600, std_ratio=2.0)
        inlier_cloud = pcd.select_by_index(ind)
        inlier_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        inlier_cloud.estimate_normals()

        inlier_cloud.orient_normals_consistent_tangent_plane(100)

        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                inlier_cloud, depth=10
            )
        densities = np.asarray(densities)
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        o3d.io.write_triangle_mesh(f"./acquisizioni/{datestring}/mesh.obj", mesh)


app.bind("<c>", lambda _: calibrate())
app.bind("<C>", lambda _: calibrate())

app.bind("<a>", lambda _: acquire())
app.bind("<A>", lambda _: acquire())

app.bind("<m>", lambda _: acquire(mesh=True))
app.bind("<M>", lambda _: acquire(mesh=True))

open_camera()

app.mainloop()
