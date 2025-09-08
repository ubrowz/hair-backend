#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 20:18:03 2025

@author: joeprous
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle
import random


app = FastAPI()

# ✅ Load model ONCE at startup
hair_segmenter_base_options = python.BaseOptions(model_asset_path="hair_segmenter.tflite")
hair_segmenter_options = vision.ImageSegmenterOptions(
    base_options=hair_segmenter_base_options,
    output_confidence_masks=True
)
hair_segmenter = vision.ImageSegmenter.create_from_options(hair_segmenter_options)

class Parameters(BaseModel):
    param1: float
    param2: float
    param3: float
    param4 : float
    param5 : float
    param6 : float
    param7 : float
    param8 : float

layers = 50
canvas_width = 180
canvas_height = 180
director = 270
gaussian = "Y"
prom = 0.15 # how much a peak should stand out from the surrounding baseline
hght = 0.35 # what should be the minimal height of a local max to be called a peak
pwr  = 30   # Suppose D is the director. pwr controls which angles around D,
            # i.e. < -d..D..d> are considered in alignment. Larger pwr values
            # mean less angles are included. See graph of function focus to
            # understand the impact of changes to pwr



def soften_mask(binary_mask, blur_size=15, sigma=5):
    """
    binary_mask: numpy array with values 0 or 255
    blur_size: size of the Gaussian kernel (odd number, e.g. 15)
    sigma: standard deviation of Gaussian
    """
    # Ensure mask is uint8
    mask = binary_mask.astype(np.uint8)

    # Apply Gaussian blur
    soft_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), sigma)

    return soft_mask


def detect_hair_with_segmenter(image, hair_segmenter):
    """Use MediaPipe Hair Segmenter to detect hair and extract color."""
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Perform hair segmentation
    segmentation_result = hair_segmenter.segment(mp_image)
    
    # Get the hair mask
    hair_mask = segmentation_result.confidence_masks[0].numpy_view()
    
    # Use a higher threshold for more precise hair detection
    binary_hair_mask = (1-(hair_mask > 0.2).astype(np.uint8)) * 255  # Increased from 0.5 to 0.7
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((6,6), np.uint8)
    binary_hair_mask = cv2.morphologyEx(binary_hair_mask, cv2.MORPH_CLOSE, kernel)
    binary_hair_mask = cv2.morphologyEx(binary_hair_mask, cv2.MORPH_OPEN, kernel)
        
    return binary_hair_mask


@app.post("/hair/")
async def segment_hair(file: UploadFile = File(...)):
    
    # Read uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Run segmentation
    hair_mask = detect_hair_with_segmenter(image, hair_segmenter)
    # Flip the binary mask. Needed because of a bug in the app.
    flipped_hair_mask = np.flipud(hair_mask)
    
    soft_flipped_hair_mask = soften_mask(flipped_hair_mask)

    # Assuming binary_hair_mask is a uint8 mask with 0 = background, 255 = hair
    # Convert to .png and return
    _, png_data = cv2.imencode(".png", soft_flipped_hair_mask)


    return Response(content=png_data.tobytes(), media_type="image/png")

def rand_coord(angle):
    xrand = canvas_width * random.random()
    yrand = canvas_height * random.random()
    if 0.0 <= angle <= 45:
        xcoord = xrand + canvas_height * math.tan(math.radians(angle))
        ycoord = yrand - canvas_height
    elif 45 < angle <= 90:
        xcoord = canvas_width + xrand
        ycoord = yrand - canvas_width * math.tan(math.radians(90 - angle))
    elif 90 < angle <= 135:
        xcoord = canvas_width + xrand
        ycoord = yrand + canvas_width * math.tan(math.radians(angle - 90))
    else:
        xcoord = xrand + canvas_height * math.tan(math.radians(180 -angle))
        ycoord = yrand + canvas_height
    return xcoord, ycoord

def rand_angle(mu, sigma):
    if sigma > 360:
        sigma = 360
    
    if gaussian == "Y":
        angle = random.gauss(mu, sigma)
    if gaussian == "N":
        angle = 360 * random.random()
    
    if angle < 0:
        angle += 360
        
    if angle > 360:
        angle -= 360
        
        
    if 180 <= angle <= 360:
        angle -= 180
    return angle

def focus(M, p):
    return abs(math.cos(math.radians(M))) ** p

def calculate_order_parameter_for_direction(angles_deg, direction_deg):
    """
    For each input fiber angle (degrees), calculate the Chebyshev order parameter
    relative to a given principal direction (degrees).
    """
    # Compute relative double angles (modulo 180 because fibers are axis-aligned)
    delta = np.deg2rad(angles_deg - direction_deg)
    cos2theta = np.cos(2 * delta)
    S = np.mean(cos2theta)  # This is the 2D "Chebyshev orientation parameter"
    return S


def peak_calculate_order_parameter_for_direction(angles_deg, direction_deg):
    """
    For each input fiber angle (degrees), calculate the Chebyshev order parameter
    relative to a given principal direction (degrees). Apply filter focus to ensure
    only nearby angles contribute to the order parameter.
    """
    # Compute relative double angles (modulo 180 because fibers are axis-aligned)
    cc = []
    delta_deg = angles_deg - direction_deg
    for angle in delta_deg:
        a = np.cos(2*angle*math.pi/180)*focus(angle, pwr)
        cc.append(a)
    S = np.mean(cc)  # This is the 2D "Hermans orientation parameter"
    return S


def peak_scan_all_directions(angles_deg):
    """
    Scan all 0...179 degrees and calculate filtered order parameter for each direction.
    """
    
    directions = np.arange(0, 180)
    S_values = []
    for direction in directions:
        S = peak_calculate_order_parameter_for_direction(angles_deg, direction)
        S_values.append(S)
    return directions, np.array(S_values)


def scan_all_directions(angles_deg):
    
    directions = np.arange(0, 180)
    S_max = -1.0
    D_max = -1.0
    S_values = []
    for direction in directions:
        S = calculate_order_parameter_for_direction(angles_deg, direction)
        if S > S_max:
            S_max = S 
            D_max = direction
        S_values.append(S)
    return D_max, S_max

@app.post("/spinner/")
async def button_clicked(params: Parameters):
    width_basic = params.param1
    nozzle_speed = params.param2
    diameter = params.param3
    rot_speed = params.param4
    flow = params.param5
    spinning_time = int(params.param6)
    voltage_level = params.param7
    hidden_constant = params.param8

        
    dep_length = math.sqrt( nozzle_speed**2 + (math.pi * diameter * (rot_speed /60) )**2)
    sigma_angle = (( hidden_constant * (flow) /((width_basic)**2)) /dep_length)/(voltage_level)
        
    whipping_factor = sigma_angle * voltage_level
    N_fibers = spinning_time * 60
   
    random.seed()

    rect_length = 150 * math.sqrt(canvas_width**2 + canvas_height**2)
    noz_angle = math.degrees(math.atan(nozzle_speed /(math.pi * diameter * (rot_speed /60))))

    director_one = director + noz_angle
    director_two = director - noz_angle
    
    currentAxis = plt.gca()
    
    fibers_in_layer = N_fibers // layers
    all_angles = []

    for i in range(N_fibers):
        
        fiber_range = i // fibers_in_layer
        fiber_color = 0.028+(0.37/layers)*fiber_range+(0.37/layers)*random.random()
        fiber_depth = layers - fiber_range
        fiber_delta_width = fiber_depth * 0.015 * width_basic
        
        
        if i % 2 == 0:
            randangle = rand_angle(director_one ,sigma_angle)
            all_angles.append(randangle)
            randcoord = rand_coord(randangle)
            vierhoek = Rectangle(randcoord ,width_basic - fiber_delta_width ,rect_length ,angle=randangle ,lw=1 ,ec="white"
                                  ,fc=str(fiber_color))

            currentAxis.add_patch(vierhoek)
        else:
            randangle = rand_angle(director_two ,sigma_angle)
            all_angles.append(randangle)
            randcoord = rand_coord(randangle)
            vierhoek = Rectangle(randcoord ,width_basic - fiber_delta_width ,rect_length ,angle=randangle ,lw=1 ,ec="white"
                                  ,fc=str(fiber_color))

            currentAxis.add_patch(vierhoek)

    currentAxis.set_aspect(aspect=1)            
    plt.xlim([0 ,canvas_width])
    plt.ylim([0 ,canvas_height])

    plt.axis("off")
    currentAxis.add_patch(plt.Rectangle((0 ,0), 1, 1, facecolor=(0 ,0 ,0) ,transform=currentAxis.transAxes, zorder=-1))
    
    primary_direction, Chybishev = scan_all_directions(all_angles)
    
    # Scan all principal directions for peaks
    directions, S_values = peak_scan_all_directions(all_angles)
    
    S_values = S_values*canvas_height
    
    plt.plot(directions, S_values, color="red")
    
    # Add text at specific coordinates
    plt.text(0, -4, f"\nWhipping Factor: {whipping_factor:.2f}\nChybishev: {Chybishev:.2f}" , fontsize=10, color="gray", ha="left", va="top")
    
    plt.text(0,-1,"0°", fontsize = 8, color="red", va="top")
    plt.text(canvas_width/2, -1,"90°", fontsize=8,color="red", ha="center",va="top")
    plt.text(canvas_width,-1,"180°", fontsize=8, color="red", ha="right",va="top")
    
    buf = io.BytesIO()
    
    plt.savefig(buf, format="png")
    buf.seek(0)

    plt.close()

    return StreamingResponse(buf, media_type="image/png")

@app.post("/flds/")
async def field_calc(params: Parameters):
    V_nozzle = params.param1
    V_rod = params.param2
    distance_nozzle_rod = params.param3
    rod_diameter = params.param4
    rod_length = params.param5
    zslice = params.param6
    ax_choice = int(params.param7)
    dummy2 = params.param8

    collector_z = 2.0           # vertical position (z-axis) of rod center

    # ===========================================
    
    # Grid for visualization
    x = np.linspace(-rod_length/2 - 1, 1+rod_length/2, 200)
    z = np.linspace(0, collector_z + distance_nozzle_rod, 400)
    X, Z = np.meshgrid(x, z)
    
    # Positions
    nozzle = (0, collector_z + distance_nozzle_rod)  # (x=0, z high up)
    collector = (0, collector_z)                     # center of rod (x=0, z=collector_z)
    
    # Collector geometry (side view: rectangle)
    collector_geometry = {
        'center': (0, collector_z),
        'width': rod_length,
        'height': rod_diameter
    }
    
    
    # ---- Distance helper for rectangle geometry ----
    def distance_to_rectangle(X, Y, center, width, height):
        """Distance from grid points to a rectangular collector (for side view)"""
        cx, cy = center
        half_w, half_h = width/2, height/2
        dx = np.maximum(0, np.maximum(cx - half_w - X, X - (cx + half_w)))
        dy = np.maximum(0, np.maximum(cy - half_h - Y, Y - (cy + half_h)))
        return np.sqrt(dx**2 + dy**2)
    
    # ---- Electric field models ----
    def electric_field_circle(nozzle_pos, collector_pos, V_nozzle, V_collector, tube_radius):
        """Approximate field for circular collector (front view, x–z plane)"""
        r_nozzle = np.sqrt((X - nozzle_pos[0])**2 + (Z - nozzle_pos[1])**2)
        r_collector = np.sqrt((X - collector_pos[0])**2 + (Z - collector_pos[1])**2)
        V = V_nozzle / (r_nozzle + 1e-6) + V_collector / (np.sqrt(r_collector**2 + tube_radius**2))
        Ex, Ez = np.gradient(-V)
        return Ex, Ez
    
    def electric_field_tube(nozzle_pos, V_nozzle, V_collector, tube_radius, collector_geom):
        """Approximate field for tube collector (side view rectangle, z–y plane)"""
        r_nozzle = np.sqrt((X - nozzle_pos[0])**2 + (Z - nozzle_pos[1])**2)
        r_collector = distance_to_rectangle(X, Z, collector_geom['center'], collector_geom['width'], collector_geom['height'])
        epsilon = 1e-10
        V = V_nozzle / (r_nozzle + 1e-6) + V_collector / (np.sqrt(r_collector**2 + epsilon))
        Ex, Ez = np.gradient(-V)
        return Ex, Ez
    
    def electric_field_top(V_nozzle, V_collector, nozzle_z, rod_z, rod_length, rod_diameter, z_slice):
        """Approximate field in x–y plane at given z_slice (top view)"""
        x = np.linspace(-rod_length/2 - 1, 1+rod_length/2, 200)
        y = np.linspace(-rod_length/2 - 1, 1+rod_length/2, 200)
        X, Y = np.meshgrid(x, y)
    
        # Distance to nozzle (point at (0,0,nozzle_z))
        r_nozzle = np.sqrt(X**2 + Y**2 + (z_slice - nozzle_z)**2)
    
        # Distance to horizontal rod segment along x-axis at y=0, z=rod_z
        # For points within rod length, use distance to line; else to nearest endpoint
        half_len = rod_length/2
        dx = np.maximum(0, np.abs(X) - half_len)
        dy = Y
        dz = z_slice - rod_z
        r_rod = np.sqrt(dx**2 + dy**2 + dz**2 + (rod_diameter/2)**2)
    
        V = V_nozzle / (r_nozzle + 1e-6) + V_collector / (r_rod + 1e-6)
        Ex, Ey = np.gradient(-V)
        return X, Y, Ex, Ey
    
    
    
    # ---- Compute fields ----
    Ex_front, Ez_front = electric_field_circle(nozzle, collector, V_nozzle, V_rod, rod_diameter/2.0)
    Ex_side, Ez_side   = electric_field_tube(nozzle, V_nozzle, V_rod, rod_diameter/2.0, collector_geometry)
    
    # ---- Plotting ----
    fig, axs = plt.subplots(1, 1, figsize=(15, 6))
    
    if ax_choice == 1:
    
        # Front view (circle rod, x–z)
        axs.set_aspect(1)
        axs.streamplot(X, Z, Ex_front, Ez_front, density=1.2, color="blue")
        axs.plot(nozzle[0], nozzle[1], 'ko', markersize=10, label="Nozzle")
        circle = plt.Circle(collector, rod_diameter/2.0, color='red', fill=False, linewidth=2, label="Rod (front view)")
        axs.add_artist(circle)
        axs.set_title("Front view (x–z)")
        axs.set_xlim(-rod_length/2 - 1, 1+rod_length/2)
        axs.set_ylim(0, collector_z + distance_nozzle_rod)
        axs.legend()
    
    if ax_choice == 2:
        # Side view (rectangle rod, z–axis vertical)
        axs.set_aspect(1)
        axs.streamplot(X, Z, Ex_side, Ez_side, density=1.2, color="blue")
        axs.plot(nozzle[0], nozzle[1], 'ko', markersize=10, label="Nozzle")
        rect = plt.Rectangle((collector_geometry['center'][0] - collector_geometry['width']/2,
                              collector_geometry['center'][1] - collector_geometry['height']/2),
                              collector_geometry['width'], collector_geometry['height'],
                              fill=True, color='red', linewidth=3, label="Rod (side view)")
        axs.add_patch(rect)
        axs.set_title("Side view (z–y)")
        axs.set_xlim(-rod_length/2 - 1, 1+rod_length/2)
        axs.set_ylim(0, collector_z + distance_nozzle_rod)
        axs.legend()
    
    if ax_choice == 3:
        
        # Top view (slice at z halfway)
        z_slice = collector_z + zslice
        
        X_top, Y_top, Ex_top, Ey_top = electric_field_top(V_nozzle, V_rod,
                                                          nozzle_z=collector_z+distance_nozzle_rod,
                                                          rod_z=collector_z,
                                                          rod_length=rod_length,
                                                          rod_diameter=rod_diameter,
                                                          z_slice=z_slice)
        axs.set_aspect(1)
        axs.streamplot(X_top, Y_top, Ex_top, Ey_top, density=1.2, color="blue")
        axs.plot(0, 0, 'ko', markersize=10, label="Nozzle projection")
        rect_top = plt.Rectangle((-rod_length/2, -rod_diameter/2),
                                 rod_length, rod_diameter,
                                 fill=True, color='red', label="Rod (top view)")
        axs.add_patch(rect_top)
        axs.set_title(f"Top view (x–y) at z ≈ {z_slice:.1f}")
        axs.set_xlim(-rod_length/2 - 1, 1+rod_length/2)
        axs.set_ylim(-rod_length/2 - 1, 1+rod_length/2)
        axs.legend()
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    
    plt.savefig(buf, format="png")

    buf.seek(0)

    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

@app.post("/multiflds/")
async def multifield_calc(params: Parameters):
    V_nozzle = params.param1
    V_rod = params.param2
    distance_nozzle_rod = params.param3
    rod_diameter = params.param4
    rod_length = params.param5
    zslice = params.param6
    ax_choice = int(params.param7)
    dummy2 = params.param8

    collector_z = 2.0  # vertical position (z-axis) of rod center
    
    # Collector geometry (side view: rectangle)
    collector_geometry = {
        'center': (0, collector_z),
        'width': rod_length,
        'height': rod_diameter
    }


    # ===========================================
    # Grid for front view (x–z)
    x = np.linspace(-rod_length/2 - 1, 1+rod_length/2, 200)
    z = np.linspace(0, collector_z + distance_nozzle_rod, 400)
    y = np.linspace(-collector_geometry['width']/2 - 1,
                     collector_geometry['width']/2 + 1, 200)

    # Multiple nozzle setup along x-axis
    n_nozzles = 5
    spacing = 1.0
    nozzle_z = collector_z + distance_nozzle_rod
    nozzle_positions = [(y0, nozzle_z) for y0 in 
                        np.linspace(-(n_nozzles-1)/2*spacing,
                                    (n_nozzles-1)/2*spacing,
                                    n_nozzles)]
    # nozzle_positions = [(0, y0, nozzle_z) for y0 in 
    #                     np.linspace(-(n_nozzles-1)/2*spacing,
    #                                 (n_nozzles-1)/2*spacing,
    #                                 n_nozzles)] 
    # # Collector
    collector = (0, collector_z)

    # ---- Helper: distance to rectangle ----
    def distance_to_rectangle(X, Y, center, width, height):
        cx, cy = center
        half_w, half_h = width/2, height/2
        dx = np.maximum(0, np.maximum(cx - half_w - X, X - (cx + half_w)))
        dy = np.maximum(0, np.maximum(cy - half_h - Y, Y - (cy + half_h)))
        return np.sqrt(dx**2 + dy**2)

    # ---- Front view field (x–z plane) ----
    def electric_field_front(nozzles, collector_pos, V_nozzle, V_collector, tube_radius):
        X, Z = np.meshgrid(x, z)
        V = np.zeros_like(X, dtype=float)
        
        for nozzle_pos in nozzles:
            r_nozzle = np.sqrt(X**2 + (Z - nozzle_pos[1])**2)
#            r_nozzle = np.sqrt((X - nozzle_pos[0])**2 + (Z - nozzle_pos[1])**2)
            V += V_nozzle / (r_nozzle + 1e-6)
        r_collector = np.sqrt((X - collector_pos[0])**2 + (Z - collector_pos[1])**2)
        V += V_collector / (np.sqrt(r_collector**2 + tube_radius**2))
        Ex, Ez = np.gradient(-V)
        return X, Z, Ex, Ez

    # ---- Side view field (y–z plane at fixed x) ----
    
    def electric_field_side(nozzles, V_nozzle, V_collector, tube_radius, collector_geom):
        # Side view grid (y–z plane)
        Y, Z = np.meshgrid(y, z)    
        V = np.zeros_like(Y, dtype=float)

        for nozzle_pos in nozzles:
            # In side view, all nozzles are projected at y=0
#            r_nozzle = np.sqrt(Y**2 + (Z - nozzle_pos[1])**2)
            r_nozzle = np.sqrt((Y - nozzle_pos[0])**2 + (Z - nozzle_pos[1])**2)
            V += V_nozzle / (r_nozzle + 1e-6)
    
        r_collector = distance_to_rectangle(Y, Z,
                                            collector_geom['center'],
                                            collector_geom['width'],
                                            collector_geom['height'])
        V += V_collector / (np.sqrt(r_collector**2 + 1e-10))
    
        Ey, Ez = np.gradient(-V)
        return Y, Z, Ey, Ez

    # ---- Top view field (x–y plane at fixed z) ----
    def electric_field_top(nozzles, V_nozzle, V_collector,
                           nozzle_z, rod_z, rod_length, rod_diameter, z_slice):
#        x = np.linspace(-rod_length/2 - 1, rod_length/2 + 1, 200)
#        y = np.linspace(-rod_length/2 - 1, rod_length/2 + 1, 200)
        X, Y = np.meshgrid(x, y)


        V = np.zeros_like(X, dtype=float)
        for nozzle_pos in nozzles:
            r_nozzle = np.sqrt((Y - nozzle_pos[0])**2 + X**2 + (z_slice - nozzle_z)**2)
            V += V_nozzle / (r_nozzle + 1e-6)

        half_len = rod_length/2
        dx = np.maximum(0, np.abs(X) - half_len)
        dy = Y
        dz = z_slice - rod_z
        r_rod = np.sqrt(dx**2 + dy**2 + dz**2 + (rod_diameter/2)**2)
        V += V_collector / (r_rod + 1e-6)

        Ex, Ey = np.gradient(-V)
        return X, Y, Ex, Ey

    # ---- Compute fields ----

    # ---- Plotting ----
    fig, axs = plt.subplots(1, 1, figsize=(15, 6))

    if ax_choice == 1:  # front
        X_side, Z_side, Ex_front, Ez_front = electric_field_front(nozzle_positions, collector, V_nozzle, V_rod, rod_diameter/2.0)
        axs.set_aspect(1)
        axs.streamplot(X_side, Z_side, Ex_front, Ez_front, density=1.2, color="blue")
        for nozzle_pos in nozzle_positions:
            axs.plot(nozzle_pos[0], nozzle_pos[1], 'ko', markersize=8)
        circle = plt.Circle(collector, rod_diameter/2.0, color='red', fill=False, linewidth=2)
        axs.add_artist(circle)
        axs.set_title("Front view (x–z)")
        axs.set_xlim(-rod_length/2 - 1, rod_length/2 + 1)
        axs.set_ylim(0, collector_z + distance_nozzle_rod)

    if ax_choice == 2:  # side
        Y_side, Z_side, Ey_side, Ez_side = electric_field_side(nozzle_positions,
                                                       V_nozzle, V_rod,
                                                       rod_diameter/2.0,
                                                       collector_geometry)    
        # Side view (rectangle rod, z–axis vertical)
        axs.set_aspect(1)
        axs.streamplot(Y_side, Z_side, Ey_side, Ez_side, density=1.2, color="blue")

        # plot nozzle projections: only those nozzles whose x_n are near x_slice will appear centered at y=0;
        # but to visualize projection of all nozzles onto this plane, you can plot them at y = 0 with a marker if desired:
        x_slice = 0.0
        for (x_n, z_n) in nozzle_positions:
            # projection of nozzle onto x=x_slice plane is at y=0; its apparent strength depends on distance (x_n - x_slice)
            if abs(x_n - x_slice) < spacing/2:   # optionally only mark near ones
                axs.plot(0.0, z_n, 'ko', markersize=8)
    
        
        # rod rectangle
        rect = plt.Rectangle((collector_geometry['center'][0] - collector_geometry['width']/2,
                              collector_geometry['center'][1] - collector_geometry['height']/2),
                              collector_geometry['width'], collector_geometry['height'],
                              fill=True, color='red', linewidth=3, label="Rod (side view)")

        # rect = plt.Rectangle((-rod_length/2, collector_z - rod_diameter/2),
        #                      rod_length, rod_diameter,
        #                      fill=True, color='red', linewidth=3)
        axs.add_patch(rect)
        axs.set_title("Side view (z–y)")
        axs.set_xlim(-rod_length/2 - 1, 1+rod_length/2)
        axs.set_ylim(0, collector_z + distance_nozzle_rod)
        axs.legend()

    if ax_choice == 3:  # top
        z_slice = collector_z + zslice
        X_top, Y_top, Ex_top, Ey_top = electric_field_top(nozzle_positions, V_nozzle, V_rod,
                                                          nozzle_z=nozzle_z,
                                                          rod_z=collector_z,
                                                          rod_length=rod_length,
                                                          rod_diameter=rod_diameter,
                                                          z_slice=z_slice)
        axs.set_aspect(1)
        axs.streamplot(X_top, Y_top, Ex_top, Ey_top, density=1.2, color="blue")
        for nozzle_pos in nozzle_positions:
            axs.plot(nozzle_pos[0], 0, 'ko', markersize=8)
        rect_top = plt.Rectangle((-rod_length/2, -rod_diameter/2),
                                 rod_length, rod_diameter,
                                 fill=True, color='red')
        axs.add_patch(rect_top)
        axs.set_title(f"Top view (x–y) at z ≈ {z_slice:.1f}")
        axs.set_xlim(-rod_length/2 - 1, rod_length/2 + 1)
        axs.set_ylim(-rod_length/2 - 1, rod_length/2 + 1)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

