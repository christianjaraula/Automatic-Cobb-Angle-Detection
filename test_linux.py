import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageTk
from ultralytics import YOLO
from typing import List, Tuple, Dict, NamedTuple
from customtkinter import CTkImage
import time

# Initialize variables
camera_active = False
cap = None
img = None
img_tensor = None
keypoints = None
outputs = None
bbox_model = None
vertebra_boxes = None
vertebra_confidences = None
show_labels = None
show_confidence = None
main_curve_frame = None
main_result_label = None
secondary_curve_box = None
curve_type_box = None 
severity_box = None
top_row_frame = None
bottom_row_frame = None


def get_kprcnn_model(path):
    try:
        num_keypoints = 4
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=(32, 64, 128, 256, 512),
            aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0),
        )

        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
            num_keypoints=num_keypoints,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
        )

        if path:
            state_dict = torch.load(path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)

        model.eval()
        return model

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return None


def initialize_models():
    global model, bbox_model, vertebra_boxes, vertebra_confidences
    bbox_path = "/home/raspi/Desktop/models/model2.pt"
    detector_path = "/home/raspi/Desktop/models/model1.pt"
    model = get_kprcnn_model(detector_path)
    bbox_model = YOLO(bbox_path)

# def initialize_models():
#     global model, bbox_model, vertebra_boxes, vertebra_confidences
#     bbox_path = "C:\\Users\\jarau\\OneDrive\\Desktop\\important\\model2.pt"
#     detector_path = "C:\\Users\\jarau\\OneDrive\\Desktop\\important\\model1.pt"
#     model = get_kprcnn_model(detector_path)
#     bbox_model = YOLO(bbox_path)

# Helper function to load and process an image
def open_image_path(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img)
    return img, img_tensor


# Filter output (convert predictions to usable format)
def filter_output(output):
    scores = output["scores"].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > 0.5)[0].tolist()
    post_nms_idxs = (
        torchvision.ops.nms(
            output["boxes"][high_scores_idxs],
            output["scores"][high_scores_idxs],
            0.3,
        )
        .cpu()
        .numpy()
    )

    np_keypoints = (
        output["keypoints"][high_scores_idxs][post_nms_idxs]
        .detach()
        .cpu()
        .numpy()
    )
    np_bboxes = (
        output["boxes"][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
    )
    np_scores = (
        output["scores"][high_scores_idxs][post_nms_idxs]
        .detach()
        .cpu()
        .numpy()
    )

    sorted_scores_idxs = np.argsort(-1 * np_scores)
    np_scores = np_scores[sorted_scores_idxs]
    np_keypoints = np_keypoints[sorted_scores_idxs]
    np_bboxes = np_bboxes[sorted_scores_idxs]

    ymins = np.array([kps[0][1] for kps in np_keypoints])
    sorted_ymin_idxs = np.argsort(ymins)

    np_scores = np.array([np_scores[idx] for idx in sorted_ymin_idxs])
    np_keypoints = np.array([np_keypoints[idx] for idx in sorted_ymin_idxs])
    np_bboxes = np.array([np_bboxes[idx] for idx in sorted_ymin_idxs])

    return np_keypoints, np_bboxes, np_scores


# Cobb angle calculation functions
def _create_angles_dict(pt, mt, tl):
    """
    pt,mt,tl: tuple(2) that contains: (angle, [idxTop, idxBottom])
    """
    return {
        "pt": {
            "angle": pt[0],
            "idxs": [pt[1][0], pt[1][1]],
        },
        "mt": {
            "angle": mt[0],
            "idxs": [mt[1][0], mt[1][1]],
        },
        "tl": {
            "angle": tl[0],
            "idxs": [tl[1][0], tl[1][1]],
        },
    }


def check_s_curve(p):
    num = len(p)
    ll = np.zeros([num - 2, 1])
    for i in range(num - 2):
        ll[i] = (p[i][1] - p[num - 1][1]) / (p[0][1] - p[num - 1][1]) - (
            p[i][0] - p[num - 1][0]
        ) / (p[0][0] - p[num - 1][0])

    flag = np.sum(np.sum(np.dot(ll, ll.T))) != np.sum(
        np.sum(abs(np.dot(ll, ll.T)))
    )
    return flag


def cobb_angle_cal(landmark_xy, image_shape):
    landmark_xy = list(landmark_xy)  # input is list
    ap_num = int(len(landmark_xy) / 2)  # number of points
    vnum = int(ap_num / 4)  # number of vertebrae

    first_half = landmark_xy[:ap_num]
    second_half = landmark_xy[ap_num:]

    # Values this function returns
    cob_angles = np.zeros(3)
    angles_with_pos = {}

    # Midpoints (2 points per vertebra) (Using a midpoint formula)
    mid_p_v = []
    for i in range(int(len(landmark_xy) / 4)):
        x = first_half[2 * i : 2 * i + 2]
        y = second_half[2 * i : 2 * i + 2]
        row = [(x[0] + x[1]) / 2, (y[0] + y[1]) / 2]
        mid_p_v.append(row)

    mid_p = []
    for i in range(int(vnum)):
        x = first_half[4 * i : 4 * i + 4]
        y = second_half[4 * i : 4 * i + 4]
        point1 = [(x[0] + x[2]) / 2, (y[0] + y[2]) / 2]
        point2 = [(x[3] + x[1]) / 2, (y[3] + y[1]) / 2]
        mid_p.append(point1)
        mid_p.append(point2)

    # Line and Slope
    vec_m = []
    for i in range(int(len(mid_p) / 2)):
        points = mid_p[2 * i : 2 * i + 2]
        row = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
        vec_m.append(row)

    mod_v = []
    for i in vec_m:
        row = [i[0] * i[0], i[1] * i[1]]
        mod_v.append(row)

    mod_v = np.sqrt(np.sum(np.matrix(mod_v), axis=1))
    dot_v = np.dot(np.matrix(vec_m), np.matrix(vec_m).T)

    angles = np.clip(dot_v / np.dot(mod_v, mod_v.T), -1, 1)
    angles = np.arccos(angles)

    maxt = np.amax(angles, axis=0)
    pos1 = np.argmax(angles, axis=0)

    pt, pos2 = np.amax(maxt), np.argmax(maxt)

    pt = pt * 180 / math.pi
    cob_angles[0] = pt

    if not check_s_curve(mid_p_v):
        mod_v1 = np.sqrt(
            np.sum(np.multiply(np.matrix(vec_m[0]), np.matrix(vec_m[0])))
        )
        mod_vs1 = np.sqrt(
            np.sum(
                np.multiply(np.matrix(vec_m[pos2]), np.matrix(vec_m[pos2])),
                axis=1,
            )
        )
        mod_v2 = np.sqrt(
            np.sum(
                np.multiply(
                    np.matrix(vec_m[int(vnum - 1)]),
                    np.matrix(vec_m[int(vnum - 1)]),
                ),
                axis=1,
            )
        )
        mod_vs2 = np.sqrt(
            np.sum(
                np.multiply(
                    vec_m[pos1.item((0, pos2))], vec_m[pos1.item((0, pos2))]
                )
            )
        )

        dot_v1 = np.dot(np.array(vec_m[0]), np.array(vec_m[pos2]).T)
        dot_v2 = np.dot(
            np.array(vec_m[int(vnum - 1)]),
            np.array(vec_m[pos1.item((0, pos2))]).T,
        )

        mt = np.arccos(np.clip(dot_v1 / np.dot(mod_v1, mod_vs1.T), -1, 1))
        tl = np.arccos(np.clip(dot_v2 / np.dot(mod_v2, mod_vs2.T), -1, 1))

        mt = mt * 180 / math.pi
        tl = tl * 180 / math.pi
        cob_angles[1] = mt
        cob_angles[2] = tl

        angles_with_pos = _create_angles_dict(
            pt=(float(pt), [pos2, pos1.A1.tolist()[pos2]]),
            mt=(
                float(mt),
                [0, int(pos2)],
            ),  # Using 0 instead of undefined pos1_1
            tl=(
                float(tl),
                [pos1.A1.tolist()[pos2], int(vnum - 1)],
            ),  # Using vnum-1 instead of undefined pos1_2
        )

    else:
        if (mid_p_v[pos2 * 2][1]) + mid_p_v[pos1.item((0, pos2)) * 2][
            1
        ] < image_shape[0]:
            # Calculate Upside Cobb Angle
            mod_v_p = np.sqrt(np.sum(np.multiply(vec_m[pos2], vec_m[pos2])))
            mod_v1 = np.sqrt(
                np.sum(np.multiply(vec_m[0:pos2], vec_m[0:pos2]), axis=1)
            )
            dot_v1 = np.dot(np.array(vec_m[pos2]), np.array(vec_m[0:pos2]).T)

            angles1 = np.arccos(
                np.clip(dot_v1 / np.dot(mod_v_p, mod_v1.T), -1, 1)
            )
            CobbAn1, pos1_1 = np.amax(angles1, axis=0), np.argmax(
                angles1, axis=0
            )
            mt = CobbAn1 * 180 / math.pi
            cob_angles[1] = mt

            # Calculate Downside Cobb Angle
            mod_v_p2 = np.sqrt(
                np.sum(
                    np.multiply(
                        vec_m[pos1.item((0, pos2))],
                        vec_m[pos1.item((0, pos2))],
                    )
                )
            )
            mod_v2 = np.sqrt(
                np.sum(
                    np.multiply(
                        vec_m[pos1.item((0, pos2)) : int(vnum)],
                        vec_m[pos1.item((0, pos2)) : int(vnum)],
                    ),
                    axis=1,
                )
            )
            dot_v2 = np.dot(
                np.array(vec_m[pos1.item((0, pos2))]),
                np.array(vec_m[pos1.item((0, pos2)) : int(vnum)]).T,
            )

            angles2 = np.arccos(
                np.clip(dot_v2 / np.dot(mod_v_p2, mod_v2.T), -1, 1)
            )
            CobbAn2, pos1_2 = np.amax(angles2, axis=0), np.argmax(
                angles2, axis=0
            )
            tl = CobbAn2 * 180 / math.pi
            cob_angles[2] = tl

            pos1_2 = pos1_2 + pos1.item((0, pos2))

            angles_with_pos = _create_angles_dict(
                mt=(float(pt), [pos2, pos1.A1.tolist()[pos2]]),
                pt=(float(mt), [int(pos1_1), int(pos2)]),
                tl=(float(tl), [pos1.A1.tolist()[pos2], int(pos1_2)]),
            )

        else:
            # Calculate Upper Upside Cobb Angle
            mod_v_p = np.sqrt(np.sum(np.multiply(vec_m[pos2], vec_m[pos2])))
            mod_v1 = np.sqrt(
                np.sum(np.multiply(vec_m[0:pos2], vec_m[0:pos2]), axis=1)
            )
            dot_v1 = np.dot(np.array(vec_m[pos2]), np.array(vec_m[0:pos2]).T)

            angles1 = np.arccos(
                np.clip(dot_v1 / np.dot(mod_v_p, mod_v1.T), -1, 1)
            )
            CobbAn1, pos1_1 = np.amax(angles1, axis=0), np.argmax(
                angles1, axis=0
            )
            mt = CobbAn1 * 180 / math.pi
            cob_angles[1] = mt

            # Calculate Upper Cobb Angle
            mod_v_p2 = np.sqrt(
                np.sum(np.multiply(vec_m[pos1_1], vec_m[pos1_1]))
            )
            mod_v2 = np.sqrt(
                np.sum(
                    np.multiply(vec_m[0 : pos1_1 + 1], vec_m[0 : pos1_1 + 1]),
                    axis=1,
                )
            )
            dot_v2 = np.dot(
                np.array(vec_m[pos1_1]), np.array(vec_m[0 : pos1_1 + 1]).T
            )

            angles2 = np.arccos(
                np.clip(dot_v2 / np.dot(mod_v_p2, mod_v2.T), -1, 1)
            )
            CobbAn2, pos1_2 = np.amax(angles2, axis=0), np.argmax(
                angles2, axis=0
            )
            tl = CobbAn2 * 180 / math.pi
            cob_angles[2] = tl

            angles_with_pos = _create_angles_dict(
                tl=(float(pt), [pos2, pos1.A1.tolist()[pos2]]),
                mt=(float(mt), [pos1_1, pos2]),
                pt=(float(tl), [int(pos1_2), int(pos1_1)]),
            )

    midpoint_lines = []
    for i in range(0, int(len(mid_p) / 2)):
        midpoint_lines.append(
            [list(map(int, mid_p[i * 2])), list(map(int, mid_p[i * 2 + 1]))]
        )

    # Remove Numpy Values
    cobb_angles_list = [float(c) for c in cob_angles]
    for key in angles_with_pos.keys():
        angles_with_pos[key]["angle"] = float(angles_with_pos[key]["angle"])
        for i in range(len(angles_with_pos[key]["idxs"])):
            angles_with_pos[key]["idxs"][i] = int(
                angles_with_pos[key]["idxs"][i]
            )

    return cobb_angles_list, angles_with_pos, midpoint_lines


def keypoints_to_landmark_xy(
    keypoints: List[List[List[float]]],
) -> List[float]:
    x_points = [kp[0] for kps in keypoints for kp in kps]
    y_points = [kp[1] for kps in keypoints for kp in kps]
    return x_points + y_points

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''GUI FOR COBB ANGLE CALCULATION'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Initialize CustomTkinter
ctk.set_appearance_mode("light")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("black-white.json")  

def update_detect_button_state():
    global img
    if img is not None:
        detect_vertebrae_button.configure(state="normal", fg_color=("#000000"))
    else:
        detect_vertebrae_button.configure(state="disabled", fg_color=("gray75", "gray45"))
        
def open_file():
    global img, img_tensor, outputs, keypoints, camera_active, cap, capture_button
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("All files", "*.*")
        ]
    )
    if not file_path:
        return

    # If camera is running, stop it
    if camera_active:
        camera_active = False
        cap.release()
        camera_button.configure(text="Camera", fg_color=("#000000"))
        # Hide the capture button when switching from camera to image
        if 'capture_button' in globals() and hasattr(capture_button, 'place_forget'):
            capture_button.place_forget()

    try:
        # # Load the original image
        # original_img, original_img_tensor = open_image_path(file_path)
        
        # # Convert to grayscale
        # gray_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY)
        
        # # Convert back to RGB format for display (but still grayscale)
        # img = cv.cvtColor(gray_img, cv.COLOR_GRAY2RGB)
        # img_tensor = F.to_tensor(img)

        img, img_tensor = open_image_path(file_path)
        
        # Reset model-related global variables
        outputs = None
        keypoints = None
        vertebra_boxes = None

        img_display = Image.fromarray(img)
        display_width = main_frame.winfo_width() - side_panel.winfo_width() - 40
        display_height = main_frame.winfo_height() - 40

        image_label.original_image = img_display
        resized_image = resize_image_for_display(img_display, display_width, display_height)

        # Convert to CTkImage
        ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))

        update_detect_button_state()

        image_label.configure(image=ctk_image)
        image_label.image = ctk_image  # Keep a reference to prevent garbage collection

        # Disable Cobb angle button until keypoints are detected
        detect_vertebrae_button.configure(state="normal", fg_color=("#000000"))
        cobb_angle_button.configure(state="disabled", fg_color=("gray75", "gray45"))
        keypoints_button.configure(state="disabled", fg_color=("gray75", "gray45"))

    except Exception as e:
        messagebox.showerror("Error", str(e))


def resize_image_for_display(image, target_width=1440, target_height=1724):
    """
    Resize image to target resolution while preserving aspect ratio.
    Adds padding when necessary to achieve exact target dimensions.
    """
    img_width, img_height = image.size
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height

    # Create a blank canvas with the target dimensions (black background)
    new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))

    if img_ratio > target_ratio:
        # Image is wider than target ratio - scale by width
        new_width = target_width
        new_height = int(new_width / img_ratio)
        resized = image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        # Center vertically
        y_offset = (target_height - new_height) // 2
        new_img.paste(resized, (0, y_offset))
    else:
        # Image is taller than target ratio - scale by height
        new_height = target_height
        new_width = int(new_height * img_ratio)
        resized = image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        # Center horizontally
        x_offset = (target_width - new_width) // 2
        new_img.paste(resized, (x_offset, 0))

    return new_img


def update_image_size(event=None):
    """Update image size when window is resized"""
    if hasattr(update_image_size, "last_width"):
        if (
            update_image_size.last_width == main_frame.winfo_width()
            and update_image_size.last_height == main_frame.winfo_height()
        ):
            return

    update_image_size.last_width = main_frame.winfo_width()
    update_image_size.last_height = main_frame.winfo_height()

    if hasattr(image_label, "original_image"):
        display_width = (
            main_frame.winfo_width() - side_panel.winfo_width() - 40
        )
        display_height = main_frame.winfo_height() - 40
        resized_image = resize_image_for_display(
            image_label.original_image, display_width, display_height
        )
        ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))

        image_label.configure(image=ctk_image)
        image_label.image = ctk_image


def create_camera_buttons():
    global capture_button
    try:
        camera_icon_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/icons8-aperture-48.png"
        pil_image = Image.open(camera_icon_path)
        camera_icon_ctk = CTkImage(light_image=pil_image, dark_image=pil_image, size=(32, 32))
        
        capture_button = ctk.CTkButton(
            image_label,
            text="",  
            image=camera_icon_ctk,
            command=capture_frame,
            width=30,
            height=30,
            corner_radius=5, 
            border_width=0,
        )
        capture_button.camera_icon_ctk = camera_icon_ctk
        
    except Exception as e:
        print(f"Error loading camera icon: {e}")
        
    capture_button.place(relx=0.5, rely=0.95, anchor="center")
    capture_button.place_forget()

def toggle_camera():
    global cap, camera_active, img, img_tensor, keypoints, outputs, capture_button, save_cobb_button
    
    if not camera_active:
        try:
            # Raspberry Pi camera setup with lower resolution for better performance
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            # Set lower resolution for better performance on Raspberry Pi
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280) #1280,640 optional
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640) #960,480 optional    
            # Set lower FPS for better performance
            cap.set(cv.CAP_PROP_FPS, 15)
            # Disable Auto White Balance (AWB)
            cap.set(cv.CAP_PROP_AUTO_WB, 0)
            # cap.set(cv.CAP_PROP_WHITE_BALANCE_BLUE_U, 5500)
            # cap.set(cv.CAP_PROP_WHITE_BALANCE_RED_V, 5500)
            
            # Set manual exposure settings
            cap.set(cv.CAP_PROP_EXPOSURE, 10000)  # Set shutter speed to 10000
            cap.set(cv.CAP_PROP_GAIN, 1.0)        # Set gain to 1.0
            
            # Optionally, fully disable auto exposure
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
            
            camera_active = True
            
            # Reset grayscale when camera is activated
            grayscale_enabled.set(True)
            # Enable grayscale switch ONLY when camera is active
            grayscale_switch.configure(state="normal")
            
            # Enable zoom slider when camera is active
            zoom_slider.configure(state="normal")
            zoom_factor.set(1.0)
            zoom_label.configure(text="1.0x")
            
            # Reset keypoints and outputs
            keypoints = None
            outputs = None
            
            # Disable the show keypoints and cobb angle buttons
            keypoints_button.configure(state="disabled", fg_color=("gray75", "gray45"))
            cobb_angle_button.configure(state="disabled", fg_color=("gray75", "gray45"))
            
            no_camera_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/icons8-no-camera-48.png"
            no_camera_pil = Image.open(no_camera_path)
            no_camera_icon_ctk = CTkImage(light_image=no_camera_pil, dark_image=no_camera_pil, size=(24, 24))
            
            camera_button.configure(
                text="Stop", 
                fg_color="#FF3B30",
                image=no_camera_icon_ctk,
                compound="left"
            )
            camera_button.image = no_camera_icon_ctk  # Keep a reference
            
            if 'save_cobb_button' in globals() and save_cobb_button is not None:
                save_cobb_button.destroy()
                save_cobb_button = None
            
            if 'capture_button' in globals() and hasattr(capture_button, 'place'):
                capture_button.place(relx=0.5, rely=0.95, anchor="center")
            else:
                create_camera_buttons()
                capture_button.place(relx=0.5, rely=0.95, anchor="center")
            
            update_camera_feed()
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")
    else:
        # Stop the camera
        camera_active = False
        cap.release()
        
        # Disable zoom slider when camera is stopped
        zoom_slider.configure(state="disabled")
        zoom_factor.set(1.0)
        zoom_label.configure(text="1.0x")
        
        # Always disable grayscale switch when camera is stopped
        grayscale_switch.configure(state="disabled")
        
        camera_button.configure(
            text="Camera", 
            fg_color=("#000000"),
            image=camera_icon_ctk  
        )
        
        if 'capture_button' in globals() and hasattr(capture_button, 'place_forget'):
            capture_button.place_forget()
        
        image_label.configure(image="")
        image_label.image = None
        detect_vertebrae_button.configure(state="disabled", fg_color=("gray75", "gray45"))

        if hasattr(image_label, 'original_image'):
            image_label.configure(image=image_label.original_image)
            
def update_zoom():
    """Update the zoom label and apply zoom in camera feed"""
    # Update the label with current zoom value
    current_zoom = zoom_factor.get()
    zoom_label.configure(text=f"{current_zoom:.1f}x")
    

def update_camera_feed():
    global img, img_tensor, camera_active

    if camera_active and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            # Flip the frame horizontally to create mirror effect
            # frame = cv.flip(frame, 1)
            
            # Apply zoom if zoom_factor is greater than 1.0
            current_zoom = zoom_factor.get()
            if current_zoom > 1.0:
                # Get frame dimensions
                h, w = frame.shape[:2]
                
                # Calculate new dimensions and offsets for zooming
                new_h, new_w = int(h / current_zoom), int(w / current_zoom)
                center_y, center_x = h // 2, w // 2
                
                # Calculate the crop region
                top = center_y - new_h // 2
                left = center_x - new_w // 2
                
                # Ensure crop region is within frame bounds
                top = max(0, top)
                left = max(0, left)
                bottom = min(h, top + new_h)
                right = min(w, left + new_w)
                
                # Crop and resize to original dimensions
                zoomed_frame = frame[top:bottom, left:right]
                frame = cv.resize(zoomed_frame, (w, h), interpolation=cv.INTER_LINEAR)
            
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Apply grayscale if enabled (only during live view)
            if grayscale_enabled.get():
                gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2RGB)
            
            # Add frame overlay
            height, width = frame.shape[:2]
            frame_w = 400  # Fixed width for frame
            frame_h = height
            frame_x = (width - frame_w) // 2
            frame_y = 0

            frame_overlay = frame.copy()
            
            # Create a very subtle medical-style frame overlay
            overlay = frame_overlay.copy()
            
            # Add extremely subtle vertical guide lines
            guide_line_color = (180, 200, 255)  # Very light medical blue
            dash_length = 8
            gap_length = 20
            
            # Draw minimal dashed vertical lines with very low opacity
            for y in range(0, height, dash_length + gap_length):
                # Left guide line
                cv.line(overlay, (frame_x, y), (frame_x, min(y + dash_length, height)), guide_line_color, 1)
                # Right guide line
                cv.line(overlay, (frame_x + frame_w, y), (frame_x + frame_w, min(y + dash_length, height)), guide_line_color, 1)
            
            # Apply very subtle overlay
            cv.addWeighted(overlay, 0.15, frame_overlay, 0.85, 0, frame_overlay)
            
            # Use frame overlay for further processing
            img = frame_overlay
            
            img_tensor = torch.from_numpy(frame_overlay.transpose(2, 0, 1)).float() / 255.0
            
            frame_display = Image.fromarray(frame_overlay)
            
            # Use full window dimensions for display
            display_width = main_frame.winfo_width() - side_panel.winfo_width()
            display_height = main_frame.winfo_height()
            
            resized_image = resize_image_for_display(frame_display, display_width, display_height)

            
            ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))
            image_label.configure(image=ctk_image)
            image_label.image = ctk_image
            
            if 'capture_button' in globals() and hasattr(capture_button, 'winfo_exists') and capture_button.winfo_exists():
                capture_button.lift()
                
            detect_vertebrae_button.configure(state="disabled", fg_color=("gray75", "gray45"))

            if camera_active:
                root.after(30, update_camera_feed)
        else:
            camera_active = False
            cap.release()
            camera_button.configure(text="Camera", fg_color=("#000000"))
            detect_vertebrae_button.configure(state="disabled", fg_color=("gray75", "gray45"))
            
            # Always disable grayscale switch when camera is stopped
            grayscale_switch.configure(state="disabled")

            if 'capture_button' in globals() and hasattr(capture_button, 'place_forget'):
                capture_button.place_forget()
        
            update_detect_button_state()
    else:
        # Camera was deactivated elsewhere
        if 'cap' in globals() and cap is not None and cap.isOpened():
            cap.release()
        camera_button.configure(text="Camera", fg_color=("#000000"))
        
        # Always disable grayscale switch when camera is stopped
        grayscale_switch.configure(state="disabled")


def capture_frame():
    global cap, camera_active, img, img_tensor
    
    if cap is not None and cap.isOpened():
        # Get current zoom factor before capture
        current_zoom = zoom_factor.get()
        # Get current grayscale setting before capture
        current_grayscale = grayscale_enabled.get()

        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280) #2560 optional
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960) # 1920 optional
        
        time.sleep(0.2)
        for _ in range(3):  
            cap.read()
        
        ret, frame = cap.read()
        if ret:
            # Apply zoom to the captured frame if zoom is greater than 1.0
            if current_zoom > 1.0:
                # Get frame dimensions
                h, w = frame.shape[:2]
                
                # Calculate new dimensions and offsets for zooming
                new_h, new_w = int(h / current_zoom), int(w / current_zoom)
                center_y, center_x = h // 2, w // 2
                
                # Calculate the crop region
                top = center_y - new_h // 2
                left = center_x - new_w // 2
                
                # Ensure crop region is within frame bounds
                top = max(0, top)
                left = max(0, left)
                bottom = min(h, top + new_h)
                right = min(w, left + new_w)
                
                # Crop and resize to original dimensions
                zoomed_frame = frame[top:bottom, left:right]
                frame = cv.resize(zoomed_frame, (w, h), interpolation=cv.INTER_LINEAR)
            
            # frame = cv.flip(frame, 1)
            # Convert BGR (OpenCV default) to RGB for proper colors
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Apply grayscale if it was enabled during capture
            if current_grayscale:
                gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2RGB)
            
            # Add frame overlay
            height, width = frame.shape[:2]
            frame_w = 400  # Fixed width for frame
            frame_h = height
            frame_x = (width - frame_w) // 2
            frame_y = 0

            frame_overlay = frame.copy()
            
            # Create a very subtle medical-style frame overlay
            overlay = frame_overlay.copy()
            
            # Add extremely subtle vertical guide lines
            guide_line_color = (180, 200, 255)  # Very light medical blue
            dash_length = 8
            gap_length = 20
            
            # Draw minimal dashed vertical lines with very low opacity
            for y in range(0, height, dash_length + gap_length):
                # Left guide line
                cv.line(overlay, (frame_x, y), (frame_x, min(y + dash_length, height)), guide_line_color, 1)
                # Right guide line
                cv.line(overlay, (frame_x + frame_w, y), (frame_x + frame_w, min(y + dash_length, height)), guide_line_color, 1)
            
            # Apply very subtle overlay
            cv.addWeighted(overlay, 0.15, frame_overlay, 0.85, 0, frame_overlay)
            
            img = frame_overlay.copy()
            
            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            
            # Update detect button state
            update_detect_button_state()
        
            if cobb_angle_button:
                cobb_angle_button.configure(state="disabled", fg_color=("gray75", "gray45"))
            if keypoints_button:
                keypoints_button.configure(state="disabled", fg_color=("gray75", "gray45"))
            
            # Display the captured frame
            img_display = Image.fromarray(img)
            display_width = main_frame.winfo_width() - side_panel.winfo_width()
            display_height = main_frame.winfo_height()
            
            # Store as original image for resizing during window changes
            image_label.original_image = img_display
            
            # Resize image for better display performance
            resized_image = resize_image_for_display(img_display, display_width, display_height)
            ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))
            
            image_label.configure(image=ctk_image)
            image_label.image = ctk_image  
            
            if detect_vertebrae_button:
                detect_vertebrae_button.configure(state="normal", fg_color=("#000000"))
            
            # Disable grayscale switch after capture as requested
            grayscale_switch.configure(state="disabled")
            
            # Disable zoom slider when image is captured
            zoom_slider.configure(state="disabled")
            
            camera_active = False
            cap.release()
            cap = None  
            
            if camera_button:
                camera_button.configure(text="Camera", fg_color=("#000000"), image=camera_icon_ctk)
            # Hide capture button if it exists
            if 'capture_button' in globals() and capture_button:
                try:
                    capture_button.place_forget()
                except AttributeError:
                    pass  
        else:
            messagebox.showerror("Error", "Failed to capture high-resolution image")

# The apply_grayscale function is now only relevant during live camera view
def apply_grayscale():
    pass
    
# def capture_frame():
#     global cap, camera_active, img, img_tensor
    
#     # Capture frame
#     if cap is not None and cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             frame = cv.flip(frame, 1)
#             frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#             img = frame
            
#             # Convert to tensor using PyTorch directly (better for Raspberry Pi than torchvision F.to_tensor)
#             img_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0

#             # Update button state when camera stops
#             update_detect_button_state()
        
#             # Disable the cobb angle button initially
#             cobb_angle_button.configure(state="disabled", fg_color=("gray75", "gray45"))
#             keypoints_button.configure(state="disabled", fg_color=("gray75", "gray45"))
            
#             # Display the captured frame as the new image
#             img_display = Image.fromarray(img)
#             display_width = main_frame.winfo_width() - side_panel.winfo_width() - 40
#             display_height = main_frame.winfo_height() - 40
            
#             # Store as original image for resizing during window changes
#             image_label.original_image = img_display
            
#             # Use BILINEAR resizing which is faster on Raspberry Pi
#             resized_image = resize_image_for_display(
#                 img_display, display_width, display_height
#             )
#             ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))
#             image_label.configure(image=ctk_image)
#             image_label.image = ctk_image
            
#             # Enable the detect button since we now have an image
#             detect_vertebrae_button.configure(state="normal", fg_color=("#000000"))
            
#             # Stop the camera after capturing the frame
#             camera_active = False
#             cap.release()
#             camera_button.configure(text="Camera", fg_color=("#000000"))
        
#             # Hide the capture button
#             if 'capture_button' in globals() and hasattr(capture_button, 'place_forget'):
#                 capture_button.place_forget()


def detect_vertebrae():
    global img, vertebra_boxes, tk_image, detection_method, vertebra_confidences, show_labels, show_confidence
    if img is None:
        messagebox.showerror(
            "Error", "No image available. Please load an image first."
        )
        return

    try:
        # Create a status label positioned over the image instead of in the main frame
        status_frame = ctk.CTkFrame(
            image_label,
            corner_radius=8,
            fg_color="#1A1A1A"
        )
        # Position the frame in the center of the image
        status_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        status_label = ctk.CTkLabel(
            status_frame,
            text="Detecting vertebrae...",
            text_color="white",
            bg_color="#1A1A1A",
            padx=20,
            pady=10
        )
        status_label.pack()
        main_frame.update()  # Force update to show the label

        # Get detection method
        method = detection_var.get()

        # Calculate frame boundaries
        height, width = img.shape[:2]
        frame_w = 400  # Fixed width for frame (must match capture_frame())
        frame_x = (width - frame_w) // 2

        # Update status with minimal UI updates
        status_label.configure(text="Running detection model...")
        main_frame.update()

        if method == "YOLO":
            img_rgb = (
                cv.cvtColor(img, cv.COLOR_BGR2RGB)
                if len(img.shape) == 2
                else img
            )
            results = bbox_model(img_rgb)
            boxes = results[0].boxes
            
            # Filter boxes to only include those within the frame
            frame_boxes = []
            frame_confidences = []
            for box, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                # Check if box's center is within the frame
                box_center_x = (box[0] + box[2]) / 2
                if frame_x <= box_center_x <= frame_x + frame_w:
                    frame_boxes.append(box)
                    frame_confidences.append(conf)
            
            vertebra_boxes = np.array(frame_boxes)
            vertebra_confidences = np.array(frame_confidences)
        else:  # Model1
            img_tensor = F.to_tensor(img)
            with torch.no_grad():
                output = model([img_tensor])[0]
            
            # Apply filter to get only high-scoring detections
            scores = output["scores"].detach().cpu().numpy()
            
            # Additional filter for frame-only detections
            high_scores_idxs = []
            filtered_boxes = []
            filtered_scores = []
            
            for idx, (box, score) in enumerate(zip(output["boxes"], output["scores"])):
                if score > 0.5:
                    # Convert box to numpy and get center x
                    np_box = box.detach().cpu().numpy()
                    box_center_x = (np_box[0] + np_box[2]) / 2
                    
                    # Check if box center is within the frame
                    if frame_x <= box_center_x <= frame_x + frame_w:
                        high_scores_idxs.append(idx)
                        filtered_boxes.append(np_box)
                        filtered_scores.append(score.detach().cpu().numpy())
            
            vertebra_boxes = np.array(filtered_boxes)
            vertebra_confidences = np.array(filtered_scores)
        
        # Update status with minimal UI updates
        status_label.configure(text="Processing detection results...")
        main_frame.update()

        # Clean up the status label before updating the display
        status_frame.destroy()
        main_frame.update()
        
        # Update the display with detected boxes
        update_vertebrae_display()
        
        # Enable keypoints button
        keypoints_button.configure(
            state="normal", fg_color=("#000000")
        )
        
    except Exception as e:
        # Clean up UI even if there's an error
        if 'status_frame' in locals() and status_frame.winfo_exists():
            status_frame.destroy()
            main_frame.update()
        messagebox.showerror("Error", f"Failed to detect vertebrae: {str(e)}")

def update_vertebrae_display():
    global img, vertebra_boxes, vertebra_confidences, show_labels, show_confidence
    
    if img is None or vertebra_boxes is None:
        return
    
    # Start with a clean copy of the original image
    img_with_boxes = img.copy()
    
    # Font settings
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    for idx, (box, conf) in enumerate(
        zip(vertebra_boxes, vertebra_confidences)
    ):
        x1, y1, x2, y2 = map(int, box[:4])

        overlay = img_with_boxes.copy()
        alpha = 0.2  

        # Draw filled rectangle with transparency
        cv.rectangle(
            overlay, (x1, y1), (x2, y2), (80, 200, 255), -1
        )  # Light blue fill
        cv.addWeighted(
            overlay, alpha, img_with_boxes, 1 - alpha, 0, img_with_boxes
        )

        # Horizontal lines (top and bottom)
        dash_length = 10
        gap_length = 5
        color = (30, 144, 255)  
        thickness = 2

        # Draw dashed lines for top and bottom borders
        for x in range(x1, x2, dash_length + gap_length):
            x_end = min(x + dash_length, x2)
            cv.line(
                img_with_boxes,
                (x, y1),
                (x_end, y1),
                color,
                thickness,
                cv.LINE_AA,
            )
            cv.line(
                img_with_boxes,
                (x, y2),
                (x_end, y2),
                color,
                thickness,
                cv.LINE_AA,
            )

        # Draw dashed lines for left and right borders
        for y in range(y1, y2, dash_length + gap_length):
            y_end = min(y + dash_length, y2)
            cv.line(
                img_with_boxes,
                (x1, y),
                (x1, y_end),
                color,
                thickness,
                cv.LINE_AA,
            )
            cv.line(
                img_with_boxes,
                (x2, y),
                (x2, y_end),
                color,
                thickness,
                cv.LINE_AA,
            )

        # Add corner highlights for emphasis (solid corners)
        corner_length = 15
        # Top-left corner
        cv.line(
            img_with_boxes,
            (x1, y1),
            (x1 + corner_length, y1),
            color,
            thickness + 1,
            cv.LINE_AA,
        )
        cv.line(
            img_with_boxes,
            (x1, y1),
            (x1, y1 + corner_length),
            color,
            thickness + 1,
            cv.LINE_AA,
        )
        # Top-right corner
        cv.line(
            img_with_boxes,
            (x2, y1),
            (x2 - corner_length, y1),
            color,
            thickness + 1,
            cv.LINE_AA,
        )
        cv.line(
            img_with_boxes,
            (x2, y1),
            (x2, y1 + corner_length),
            color,
            thickness + 1,
            cv.LINE_AA,
        )
        # Bottom-left corner
        cv.line(
            img_with_boxes,
            (x1, y2),
            (x1 + corner_length, y2),
            color,
            thickness + 1,
            cv.LINE_AA,
        )
        cv.line(
            img_with_boxes,
            (x1, y2),
            (x1, y2 - corner_length),
            color,
            thickness + 1,
            cv.LINE_AA,
        )
        # Bottom-right corner
        cv.line(
            img_with_boxes,
            (x2, y2),
            (x2 - corner_length, y2),
            color,
            thickness + 1,
            cv.LINE_AA,
        )
        cv.line(
            img_with_boxes,
            (x2, y2),
            (x2, y2 - corner_length),
            color,
            thickness + 1,
            cv.LINE_AA,
        )

        # Prepare label text based on checkbox selections
        label_parts = []
        if show_labels.get():
            label_parts.append("Vertebra")
        if show_confidence.get():
            label_parts.append(f"{conf:.2f}")

        # Only proceed with label if there's something to show
        if label_parts:
            label = ": ".join(label_parts)

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv.rectangle(
                img_with_boxes,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                (0, 76, 153),
                -1,
            )  # Filled rectangle

            # Draw text with black color for better visibility
            cv.putText(
                img_with_boxes,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                font_thickness,
            )

    # Update display
    img_display = Image.fromarray(img_with_boxes)
    display_width = (
        main_frame.winfo_width() - side_panel.winfo_width() - 40
    )
    display_height = main_frame.winfo_height() - 40

    image_label.original_image = img_display
    resized_image = resize_image_for_display(
        img_display, display_width, display_height
    )
    ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))
    image_label.configure(image=ctk_image)
    image_label.image = ctk_image


def show_keypoints():
    try:
        global img, keypoints, tk_image, vertebra_boxes
        if img is None:
            messagebox.showerror(
                "Error",
                "No image available. Please load an image or capture from camera first.",
            )
            return

        if vertebra_boxes is None:
            messagebox.showerror(
                "Error",
                "Please detect vertebrae first using the 'Detect Vertebrae' button.",
            )
            return

        # Create a status frame positioned over the image
        status_frame = ctk.CTkFrame(
            image_label,
            corner_radius=8,
            fg_color="#1A1A1A"
        )
        # Position the frame in the center of the image
        status_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        status_label = ctk.CTkLabel(
            status_frame,
            text="Detecting keypoints...",
            text_color="white",
            bg_color="#1A1A1A",
            padx=20,
            pady=10
        )
        status_label.pack()
        main_frame.update()  # Force update to show the label

        # Update status with minimal UI updates
        status_label.configure(text="Running keypoint model...")
        main_frame.update()

        # Run keypoint detection on full image
        img_tensor = F.to_tensor(img)
        with torch.no_grad():
            output = model([img_tensor])[0]

        kpts, _, _ = filter_output(output)

        # Update status
        status_label.configure(text="Filtering and visualizing keypoints...")
        main_frame.update()
        # Create a clean copy of the image for visualization
        img_with_detections = img.copy()
        filtered_keypoints = []

        # Calculate scaling factor based on image resolution
        height, width = img_with_detections.shape[:2]
        base_resolution = 1000  # Base reference resolution
        resolution_factor = min(height, width) / base_resolution

        # Define fixed sizes for keypoint visualization
        outer_radius = int(10 * resolution_factor)
        middle_radius = int(7 * resolution_factor)
        inner_radius = int(4 * resolution_factor)
        line_thickness = max(1, int(2 * resolution_factor))

        if len(kpts) > 0:
            # For each set of keypoints
            for keypoint_set in kpts:
                points_in_boxes = 0

                # Check each point against all boxes
                for point in keypoint_set:
                    point_in_any_box = False
                    x, y = point[0], point[1]

                    # Check against each box
                    for box in vertebra_boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        pad = 20
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(img.shape[1], x2 + pad)
                        y2 = min(img.shape[0], y2 + pad)

                        if x1 <= x <= x2 and y1 <= y <= y2:
                            point_in_any_box = True
                            break

                    if point_in_any_box:
                        points_in_boxes += 1

                # Only keep keypoint set if all 4 points are near boxes
                if points_in_boxes == 4:
                    filtered_keypoints.append(keypoint_set)

                    # Create enhanced visualization for each keypoint with fixed size relative to resolution
                    for point in keypoint_set:
                        x, y = int(point[0]), int(point[1])

                        # Outer ring - medical blue
                        cv.circle(
                            img_with_detections,
                            (x, y),
                            outer_radius,
                            (209, 134, 0),
                            line_thickness,
                            cv.LINE_AA,
                        )

                        # Middle ring - white for contrast
                        cv.circle(
                            img_with_detections,
                            (x, y),
                            middle_radius,
                            (255, 255, 255),
                            line_thickness,
                            cv.LINE_AA,
                        )

                        # Inner filled circle - light blue
                        cv.circle(
                            img_with_detections,
                            (x, y),
                            inner_radius,
                            (235, 206, 135),
                            -1,
                            cv.LINE_AA,
                        )

        # Clean up the status label before updating the display
        status_frame.destroy()
        main_frame.update()

        # Update global keypoints
        if filtered_keypoints:
            keypoints = np.array(filtered_keypoints)

            # Display result
            img_display = Image.fromarray(img_with_detections)
            display_width = (
                main_frame.winfo_width() - side_panel.winfo_width() - 40
            )
            display_height = main_frame.winfo_height() - 40

            image_label.original_image = img_display
            resized_image = resize_image_for_display(
                img_display, display_width, display_height
            )
            ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))
            image_label.configure(image=ctk_image)
            image_label.image = ctk_image
            cobb_angle_button.configure(
                state="normal", fg_color=("#000000")
            )

        else:
            messagebox.showerror(
                "Error",
                "No valid keypoints were detected near vertebrae regions.",
            )

    except Exception as e:
        if 'status_frame' in locals() and status_frame.winfo_exists():
            status_frame.destroy()
            main_frame.update()
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def apply_cobb_angle():
    try:
        # Update global variables list
        global img, keypoints, cobb_results, mid_points, angles_with_pos, cobb_angles, img_with_cobb
        global top_row_frame, bottom_row_frame
        global main_curve_frame, main_result_label
        global secondary_curve_frame, secondary_result_label
        global curve_type_frame, curve_type_result_label
        global severity_frame, severity_result_label
        # Validate image and keypoints are available
        if img is None or keypoints is None:
            messagebox.showerror(
                "Error",
                "No image or keypoints available. Please load an image first.",
            )
            return None

        landmark_xy = keypoints_to_landmark_xy(keypoints)
        cobb_results = cobb_angle_cal(landmark_xy, img.shape)
        cobb_angles, angles_with_pos, mid_points = cobb_results

        # Process each curve and identify its anatomical type
        total_vertebrae = len(mid_points)
        angle_data = []

        # Get all angles and their data
        for curve_type, data in angles_with_pos.items():
            angle = abs(data["angle"])
            top_idx = data["idxs"][0]
            bot_idx = data["idxs"][1]
            angle_data.append((angle, curve_type, data))

        # Sort angles by magnitude
        sorted_angles = sorted(angle_data, key=lambda x: x[0], reverse=True)

        # Get the main and compensation curves
        main_angle = sorted_angles[0][0]  
        secondary_angle = sorted_angles[1][0] 

        # Severity Classification
        def classify_severity(angle):
            if angle < 10:
                return "No Scoliosis"
            elif 10 <= angle <= 24:
                return "Mild"
            elif 25 <= angle <= 39:
                return "Moderate"
            else:
                return "Severe"

        # Determine curve type
        # Convert mid_points to the format expected by check_s_curve function
        mid_p_v = []
        for mp_line in mid_points:
            mid_x = (mp_line[0][0] + mp_line[1][0]) / 2
            mid_y = (mp_line[0][1] + mp_line[1][1]) / 2
            mid_p_v.append([mid_x, mid_y])

        # Check if the curve is S-type
        is_s_curve = check_s_curve(mid_p_v)

        # First check if it's "No Scoliosis" based on the main angle
        if main_angle < 10:
            curve_type = "Normal"
        else:
            # If there is scoliosis, then determine if it's S or C type
            curve_type = "S" if is_s_curve else "C"

        severity = classify_severity(main_angle)

        # Update result text for side display
        curve_type_result_label.configure(text=f"{curve_type}")
        main_result_label.configure(text=f"{main_angle:.2f}°")
        secondary_result_label.configure(text=f"{secondary_angle:.2f}°")
        severity_result_label.configure(text=f"{severity}")

        # Draw visualization
        img_with_cobb = img.copy()
        height, width = img_with_cobb.shape[:2]
        base_resolution = 1000  # Base reference resolution
        resolution_factor = min(height, width) / base_resolution

        # Define fixed sizes for visual elements that will be scaled
        circle_radius_outer = int(12 * resolution_factor)
        circle_radius_middle = int(6 * resolution_factor)
        circle_radius_inner = int(3 * resolution_factor)
        line_thickness_base = max(2, int(3 * resolution_factor))

        def extend_line(p1, p2, height, width):
            """Extend a line defined by two points to the image boundaries"""
            x1, y1 = p1
            x2, y2 = p2

            if x2 - x1 == 0:  # Vertical line
                return [(x1, 0), (x1, height)]

            slope = (y2 - y1) / (x2 - x1)
            b = y1 - slope * x1

            x_left = 0
            y_left = int(slope * x_left + b)

            x_right = width - 1
            y_right = int(slope * x_right + b)

            y_top = 0
            x_top = int((y_top - b) / slope) if slope != 0 else x1

            y_bottom = height - 1
            x_bottom = int((y_bottom - b) / slope) if slope != 0 else x1

            points = []
            if 0 <= y_left < height:
                points.append((x_left, y_left))
            if 0 <= y_right < height:
                points.append((x_right, y_right))
            if 0 <= x_top < width:
                points.append((x_top, y_top))
            if 0 <= x_bottom < width:
                points.append((x_bottom, y_bottom))

            points.sort()
            return [points[0], points[-1]]

        def get_line_midpoint(p1, p2):
            """Calculate midpoint of a line"""
            return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        def add_angle_label(img, text, position, color=(255, 255, 255)):
            """Add angle label with medical-style background box with fixed sizes"""
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, 0.8 * resolution_factor)
            thickness = max(1, int(2 * resolution_factor))
            padding = int(8 * resolution_factor)

            (text_width, text_height), baseline = cv.getTextSize(
                text, font, font_scale, thickness
            )

            box_x1 = position[0] - padding
            box_y1 = position[1] - text_height - padding
            box_x2 = position[0] + text_width + padding
            box_y2 = position[1] + padding

            # Create a semi-transparent background with rounded appearance
            overlay = img.copy()
            cv.rectangle(
                overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 60, 120), -1
            )
            cv.addWeighted(overlay, 0.7, img, 0.3, 0, img)

            # Add a subtle border
            border_thickness = max(1, int(1 * resolution_factor))
            cv.rectangle(
                img,
                (box_x1, box_y1),
                (box_x2, box_y2),
                (0, 120, 215),
                border_thickness,
                cv.LINE_AA,
            )

            # Add text with a slight shadow effect for better readability
            cv.putText(
                img,
                text,
                (position[0] + 1, position[1] + 1),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv.LINE_AA,
            )
            cv.putText(
                img,
                text,
                position,
                font,
                font_scale,
                color,
                thickness,
                cv.LINE_AA,
            )

        # Draw circles at midpoints instead of crosses
        for mp_line in mid_points:
            mid_x = int((mp_line[0][0] + mp_line[1][0]) / 2)
            mid_y = int((mp_line[0][1] + mp_line[1][1]) / 2)

            # Outer glow effect
            cv.circle(
                img_with_cobb,
                (mid_x, mid_y),
                circle_radius_outer,
                (0, 175, 255),
                max(1, int(2 * resolution_factor)),
                cv.LINE_AA,
            )
            # Inner filled circle
            cv.circle(
                img_with_cobb,
                (mid_x, mid_y),
                circle_radius_middle,
                (255, 255, 255),
                -1,
                cv.LINE_AA,
            )
            # Center dot
            cv.circle(
                img_with_cobb,
                (mid_x, mid_y),
                circle_radius_inner,
                (0, 120, 215),
                -1,
                cv.LINE_AA,
            )

        # Draw lines and labels for the two largest angles
        for angle_info in sorted_angles[:2]:
            magnitude, curve_type, angle_data = angle_info
            top, bot = angle_data["idxs"]

            # Draw the lines
            lines = []
            colors = [(0, 120, 215)]  # Bright blue

            for idx, points in enumerate(
                [
                    (mid_points[top][0], mid_points[top][1]),
                    (mid_points[bot][0], mid_points[bot][1]),
                ]
            ):
                extended = extend_line(points[0], points[1], height, width)
                # Create a gradient effect with fixed thickness regardless of resolution
                base_thickness = line_thickness_base
                for i in range(3):
                    thickness = (
                        base_thickness - i if base_thickness - i > 0 else 1
                    )
                    alpha = max(0.3, 1 - (i * 0.15))
                    color = colors[idx % len(colors)]

                    cv.line(
                        img_with_cobb,
                        tuple(extended[0]),
                        tuple(extended[1]),
                        color,
                        thickness,
                        cv.LINE_AA,
                    )
                lines.append(extended)

            # Add angle label
            midpoint = get_line_midpoint(
                get_line_midpoint(lines[0][0], lines[0][1]),
                get_line_midpoint(lines[1][0], lines[1][1]),
            )

            label = f"{magnitude:.2f} degrees"
            add_angle_label(img_with_cobb, label, midpoint)

        # Display the image
        img_display = Image.fromarray(img_with_cobb)
        display_width = (
            main_frame.winfo_width() - side_panel.winfo_width() - 40
        )
        display_height = main_frame.winfo_height() - 40

        image_label.original_image = img_display
        resized_image = resize_image_for_display(
            img_display, display_width, display_height
        )
        ctk_image = CTkImage(light_image=resized_image, size=(display_width, display_height))
        image_label.configure(image=ctk_image)
        image_label.image = ctk_image
        add_save_button()
        return img_with_cobb
    except Exception as e:
        messagebox.showerror("Error", str(e))


def save_detected_image():
    global img, img_with_cobb

    if img is None:
        messagebox.showerror("Error", "No image available to save.")
        return
    
    if img_with_cobb is None:
        img_with_cobb = apply_cobb_angle()
        if img_with_cobb is None:
            return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
    )
    
    if file_path:
        try:
            # Check if image is already in RGB format
            if len(img_with_cobb.shape) == 3 and img_with_cobb.shape[2] == 3:  
                if np.mean(img_with_cobb[:, :, 0]) > np.mean(img_with_cobb[:, :, 2]):  
                    img_to_save = cv.cvtColor(img_with_cobb, cv.COLOR_BGR2RGB)  
                else:
                    img_to_save = img_with_cobb  
            else:
                img_to_save = img_with_cobb  

            pil_image = Image.fromarray(img_to_save)
            pil_image.save(file_path)

            messagebox.showinfo("Success", f"Image with Cobb angle saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")

def add_save_button():
    global save_cobb_button

    if 'save_cobb_button' in globals() and save_cobb_button is not None:
        save_cobb_button.destroy()
        save_cobb_button = None

    if not camera_active and img_with_cobb is not None:
        save_icon_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/save.png"
        pil_image = Image.open(save_icon_path)
        save_icon = CTkImage(light_image=pil_image, size=(32, 32))

        # Create button with the icon
        save_cobb_button = ctk.CTkButton(
            image_label,
            image=save_icon,
            text="",  
            command=save_detected_image,
            hover_color="lightgray",
            width=30,
            height=30
        )
  
        # Position the button in the lower right corner
        save_cobb_button.place(
            relx=0.95,  
            rely=0.95,  
            anchor=tk.SE
        )

        save_cobb_button.icon = save_icon

# Initialize models
initialize_models()

def on_escape(event):
    if root.state() == "zoomed":
        root.state("normal")
    else:
        root.state("zoomed")

root = ctk.CTk()
root.title("Cobb Angle Calculation")
root.geometry("1920x1080")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Bind the Escape key to the on_escape function
root.bind("<Escape>", on_escape)

# Main frame with grid
main_frame = ctk.CTkFrame(root)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)  

# Side panel - increased width to 300
side_panel = ctk.CTkFrame(main_frame, width=280)
side_panel.grid(row=0, column=0, sticky="ns")
side_panel.grid_propagate(False)  
side_panel.pack_propagate(False)

button_width = 260

# Image label
image_label = ctk.CTkLabel(main_frame, text="")
image_label.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

create_camera_buttons()

def on_closing():
    global camera_active
    if camera_active:
        camera_active = False
        cap.release()
    root.destroy()

box_size = min(button_width//1 - 1, 110)  

# Create a frame for the top row (Main and Secondary)
top_row_frame = ctk.CTkFrame(
    side_panel, 
    fg_color="transparent"
)
top_row_frame.pack(side="top", fill="x", pady=(20, 10), padx=20)

# Main curve frame (top left) 
main_curve_frame = ctk.CTkFrame(
    top_row_frame,
    width=box_size,
    height=box_size,
    corner_radius=10,
    fg_color=("#F3F3F7", "#2c2c2c"),
    border_color=("#D1D1D9", "#2c2c2c"),
    border_width=2,
)
main_curve_frame.pack(side="left", padx=(0, 10))
main_curve_frame.pack_propagate(False)  

# Add the icon to the main curve frame
main_icon_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/angle-90.png"
main_icon_image = Image.open(main_icon_path)
main_icon_ctk = CTkImage(light_image=main_icon_image, dark_image=main_icon_image, size=(24, 24))
main_icon_label = ctk.CTkLabel(
    main_curve_frame,
    image=main_icon_ctk,
    text="",
)
main_icon_label.pack(pady=(20, 0))  

# Add the result text below the icon
main_result_label = ctk.CTkLabel(
    main_curve_frame,
    text="0.0°",
    font=("Arial", 14, "bold"),
    text_color=("black", "white"),
)
main_result_label.pack(pady=(5, 0))

# Add subtitle text
main_subtitle_label = ctk.CTkLabel(
    main_curve_frame,
    text="Main",
    font=("Arial", 10),
    text_color=("gray50", "gray70"),
)
main_subtitle_label.pack(pady=(0, 2))

# Secondary curve frame (top right) - replacing the text box with a frame
secondary_curve_frame = ctk.CTkFrame(
    top_row_frame,
    width=box_size,
    height=box_size,
    corner_radius=10,
    fg_color=("#F3F3F7", "#2c2c2c"),
    border_color=("#D1D1D9", "#2c2c2c"),
    border_width=2,
)
secondary_curve_frame.pack(side="right", padx=(10, 0))
secondary_curve_frame.pack_propagate(False)  

# Add the icon to the secondary curve frame
secondary_icon_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/angle.png"
secondary_icon_image = Image.open(secondary_icon_path)
secondary_icon_ctk = CTkImage(light_image=secondary_icon_image, dark_image=secondary_icon_image, size=(24, 24))
secondary_icon_label = ctk.CTkLabel(
    secondary_curve_frame,
    image=secondary_icon_ctk,
    text="",
)
secondary_icon_label.pack(pady=(20, 0))  

# Add the result text below the icon
secondary_result_label = ctk.CTkLabel(
    secondary_curve_frame,
    text="0.0°",
    font=("Arial", 14, "bold"),
    text_color=("black", "white"),
)
secondary_result_label.pack(pady=(5, 0))

secondary_subtitle_label = ctk.CTkLabel(
    secondary_curve_frame,
    text="Secondary",
    font=("Arial", 10),
    text_color=("gray50", "gray70"),
)
secondary_subtitle_label.pack(pady=(0, 3))

# Create a frame for the bottom row (Curve Type and Severity)
bottom_row_frame = ctk.CTkFrame(
    side_panel, 
    fg_color="transparent"
)
bottom_row_frame.pack(side="top", fill="x", pady=(5, 10), padx=20)

# Curve type frame (bottom left) 
curve_type_frame = ctk.CTkFrame(
    bottom_row_frame,
    width=box_size,
    height=box_size,
    corner_radius=10,
    fg_color=("#F3F3F7", "#2c2c2c"),
    border_color=("#D1D1D9", "#2c2c2c"),
    border_width=2,
)
curve_type_frame.pack(side="left", padx=(0, 10))
curve_type_frame.pack_propagate(False)  

# Add the icon to the curve type frame
curve_type_icon_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/scoliosis.png"
curve_type_icon_image = Image.open(curve_type_icon_path)
curve_type_icon_ctk = CTkImage(light_image=curve_type_icon_image, dark_image=curve_type_icon_image, size=(24, 24))
curve_type_icon_label = ctk.CTkLabel(
    curve_type_frame,
    image=curve_type_icon_ctk,
    text="",
)
curve_type_icon_label.pack(pady=(20, 0))  

# Add the result text below the icon
curve_type_result_label = ctk.CTkLabel(
    curve_type_frame,
    text="-",
    font=("Arial", 14, "bold"),
    text_color=("black", "white"),
)
curve_type_result_label.pack(pady=(1, 0))

curve_subtitle_label = ctk.CTkLabel(
    curve_type_frame,
    text="Curve Type",
    font=("Arial", 10),
    text_color=("gray50", "gray70"),
)
curve_subtitle_label.pack(pady=(0, 3))

# Severity frame (bottom right) 
severity_frame = ctk.CTkFrame(
    bottom_row_frame,
    width=box_size,
    height=box_size,
    corner_radius=10,
    fg_color=("#F3F3F7", "#2c2c2c"),
    border_color=("#D1D1D9", "#2c2c2c"),
    border_width=2,
)
severity_frame.pack(side="right", padx=(10, 0))
severity_frame.pack_propagate(False)  

# Add the icon to the severity frame
severity_icon_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/rating.png"
severity_icon_image = Image.open(severity_icon_path)
severity_icon_ctk = CTkImage(light_image=severity_icon_image, dark_image=severity_icon_image, size=(24, 24))
severity_icon_label = ctk.CTkLabel(
    severity_frame,
    image=severity_icon_ctk,
    text="",
)
severity_icon_label.pack(pady=(20, 0))  

# Add the result text below the icon
severity_result_label = ctk.CTkLabel(
    severity_frame,
    text="-",
    font=("Arial", 14, "bold"),
    text_color=("black", "white"),
)
severity_result_label.pack(pady=(1, 0))

severity_subtitle_label = ctk.CTkLabel(
    severity_frame,
    text="Severity",
    font=("Arial", 10),
    text_color=("gray50", "gray70"),
)
severity_subtitle_label.pack(pady=(0, 3))

#Process buttons
measure_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/ruler.png"
measure_icon_image = Image.open(measure_path)
measure_icon_ctk = CTkImage(light_image=measure_icon_image, dark_image=measure_icon_image, size=(24, 24))
cobb_angle_button = ctk.CTkButton(
    side_panel,
    text="Measure",
    image=measure_icon_ctk,
    compound="left",
    command=apply_cobb_angle,
    corner_radius=10,
    width=260,
    state="disabled", 
    fg_color=("gray75", "gray45"),  
    height=60
)
cobb_angle_button.pack(side="bottom", pady=(5, 20), padx=20)

target_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/points.png"
target_icon_image = Image.open(target_path)
target_icon_ctk = CTkImage(light_image=target_icon_image, dark_image=target_icon_image, size=(24, 24))
keypoints_button = ctk.CTkButton(
    side_panel,
    image=target_icon_ctk,
    compound="left",
    text="Show",
    command=show_keypoints,
    corner_radius=10,
    state="disabled", 
    fg_color=("gray75", "gray45"),  
    width=button_width,
    anchor="center",  
    height=60
)
keypoints_button.pack(side="bottom", pady=(5, 5), padx=20)

bone_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/bounding-box.png"
bone_icon_image = Image.open(bone_path)
bone_icon_ctk = CTkImage(light_image=bone_icon_image, dark_image=bone_icon_image, size=(24, 24))
detect_vertebrae_button = ctk.CTkButton(
    side_panel,
    text="Detect",
    image=bone_icon_ctk,
    compound="left",
    command=detect_vertebrae,
    state="disabled",  
    fg_color=("gray75", "gray45"),
    corner_radius=10,
    width=button_width,
    height=60
)
detect_vertebrae_button.pack(side="bottom", pady=(5, 5), padx=20)

#camera and image buttons
button_frame = ctk.CTkFrame(side_panel, fg_color="transparent") 
button_frame.pack(side="bottom", pady=(5, 5), padx=20)

camera_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/camera.png"
camera_icon_image = Image.open(camera_path)
camera_icon_ctk = CTkImage(light_image=camera_icon_image, dark_image=camera_icon_image, size=(24, 24))
camera_button = ctk.CTkButton(
    button_frame,
    text="Camera",
    image=camera_icon_ctk,
    compound="left",
    command=toggle_camera,
    corner_radius=10,
    width=115,  
    height=60,
    anchor="center"
)
camera_button.pack(side="left", padx=(0, 10))  
camera_button.image = camera_icon_ctk 

image_path = "/home/raspi/Desktop/test/Automatic-Cobb-Angle-Detection/picture.png"
image_icon_image = Image.open(image_path)
image_icon_ctk = CTkImage(light_image=image_icon_image, dark_image=image_icon_image, size=(24, 24))
open_button = ctk.CTkButton(
    button_frame,
    text="Image",
    image=image_icon_ctk,
    compound="left",
    command=open_file,
    corner_radius=10,
    width=150,  
    height=60,
    anchor="center"
)
open_button.pack(side="left")
open_button.image = image_icon_ctk

# Create radio buttons for detection method
detection_var = tk.StringVar(value="Model1") 
detection_label = ctk.CTkLabel(
    side_panel, text="Detection Method:", anchor="w",
    text_color=("gray50", "gray70"), font=("Arial", 12)
)
detection_label.pack(side="top", pady=(10, 10), padx=20)

# Create a frame to hold the radio buttons vertically
radio_frame = ctk.CTkFrame(side_panel, fg_color="transparent")
radio_frame.pack(side="top", pady=10, padx=20, fill="x")

# Add the radio buttons to the frame vertically
model1_radio = ctk.CTkRadioButton(
    radio_frame,
    text="Model 1",
    variable=detection_var,
    value="Model1",
)
model1_radio.pack(side="top", pady=(0, 20), anchor="w")  # Changed to "top" for vertical stacking

model2_radio = ctk.CTkRadioButton(
    radio_frame,
    text="Model 2",
    variable=detection_var,
    value="YOLO",
)
model2_radio.pack(side="top", anchor="w")  # Changed to "top" for vertical stacking

display_options_label = ctk.CTkLabel(
    side_panel, text="Display Options:", anchor="w",
    text_color=("gray50", "gray70"), font=("Arial", 12)
)
display_options_label.pack(side="top", pady=(10, 10), padx=20)

# Create BooleanVar for checkboxes
show_labels = ctk.BooleanVar(value=False)
show_confidence = ctk.BooleanVar(value=False)

labels_switch = ctk.CTkSwitch(
    side_panel,
    text="Show Labels",
    variable=show_labels,
    width=button_width // 1,
    command=lambda: update_vertebrae_display() if vertebra_boxes is not None else None,
)
labels_switch.pack(side="top", pady=10, padx=20)

confidence_switch = ctk.CTkSwitch(
    side_panel,
    text="Show Confidence",
    variable=show_confidence,
    width=button_width // 1,
    command=lambda: update_vertebrae_display() if vertebra_boxes is not None else None,
)
confidence_switch.pack(side="top", pady=10, padx=20)

filter_option = ctk.CTkLabel(
    side_panel, text="Camera Options:", anchor="w",
    text_color=("gray50", "gray70"), font=("Arial", 12)
)
filter_option.pack(side="top", pady=(10, 10), padx=20)
# Add this with your other global variable declarations
grayscale_enabled = ctk.BooleanVar(value=True)  # Default to True

# Update the grayscale switch creation
grayscale_switch = ctk.CTkSwitch(
    side_panel,
    text="Grayscale Mode",
    variable=grayscale_enabled,
    width=button_width // 1,
    command=lambda: apply_grayscale() if img is not None else None,
    state="disabled"  # Start disabled
)
grayscale_switch.pack(side="top", pady=10, padx=20)

# Define a global variable for zoom factor
zoom_factor = ctk.DoubleVar(value=1.0)

# Create the zoom slider
zoom_slider = ctk.CTkSlider(
    side_panel,
    from_=1.0,
    to=3.0,
    number_of_steps=20,
    variable=zoom_factor,
    width=button_width // 1,
    command=lambda value: update_zoom(),
    state="disabled"  # Start disabled
)
zoom_slider.pack(side="top", pady=10, padx=20)

# Add a label for the zoom slider
zoom_label = ctk.CTkLabel(
    side_panel,
    text="1.0x",
    font=("Helvetica", 10)
)
zoom_label.pack(side="top", pady=(0,1), padx=20)

spacer_bottom = ctk.CTkLabel(side_panel, text="", height=50)
spacer_bottom.pack(side="bottom")


def on_closing():
    global camera_active
    if camera_active:
        camera_active = False
        cap.release()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()