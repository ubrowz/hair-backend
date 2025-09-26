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
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import random
import re
import sys
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable



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
    param4: float  
    param5: float  
    param6: float  
    param7: float    
    param8: float   
    param9: float
    param10: float
    param11: float
    param12: float
    param13: float
    param14: float
    param15: float
    param16: float
    param17: float
    param18: float
    param19: float


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

from scipy.stats.norm import pdf

@app.post("/multiflds/")
async def multifield_calc(params: Parameters):
    

    # Electric field (nozzle + rod + plates)
    # def electric_field_needle(x, y, z):
        
    #     # Start with zero field
    #     Ex, Ey, Ez = 0.0, 0.0, 0.0
        
    #     # --- Contribution from each nozzle (finite needle) ---
    #     N_seg_nozzle = 10
    #     for (xn, yn, zn) in nozzle_positions:
    #         # Discretize nozzle as a line in y-direction
    #         zs = np.linspace(zn, zn + nozzle_length, N_seg_nozzle)
    #         for zi in zs:
    #             rx, ry, rz = (x - xn), (y - yn), (z - zi)
    #             r3 = (rx**2 + ry**2 + rz**2 + 1e-9)**1.5
    #             Ex += V_nozzle * rx / r3
    #             Ey += V_nozzle * ry / r3
    #             Ez += V_nozzle * rz / r3
        
    #     # --- Contribution from the rod (cylinder along x-axis) ---
    #     N_seg = 50
    #     xs = np.linspace(-rod_length/2, rod_length/2, N_seg)
    #     Ex_rod = Ey_rod = Ez_rod = 0.0
    #     for xi in xs:
    #         r_seg = np.sqrt(((x - xi))**2 + (y)**2 + ((z - rod_z))**2)
    #         Ex_rod += V_rod * ((x - xi)) / (r_seg**3 + 1e-9)
    #         Ey_rod += V_rod * (y) / (r_seg**3 + 1e-9)
    #         Ez_rod += V_rod * ((z - rod_z)) / (r_seg**3 + 1e-9)
            
    #     # --- Contribution from plates ---
    #     def plate_field(x, y, z, plate_center, V_plate):
    #         xp, yp, zp = plate_center
    #         Ny, Nz = 10, 10  # discretization resolution
    #         ys = np.linspace(yp - plate_height/2, yp + plate_height/2, Ny)
    #         zs = np.linspace(zp - plate_width/2, zp + plate_width/2, Nz)
    #         Ex_p, Ey_p, Ez_p = 0.0, 0.0, 0.0
    #         for yi in ys:
    #             for zi in zs:
    #                 rx, ry, rz = (x - xp), (y - yi), (z - zi)
    #                 r3 = (rx**2 + ry**2 + rz**2 + 1e-9)**1.5
    #                 Ex_p += V_plate * rx / r3
    #                 Ey_p += V_plate * ry / r3
    #                 Ez_p += V_plate * rz / r3
    #         return Ex_p, Ey_p, Ez_p
       
    #     plate1_center = plate1_position[0]
    #     plate2_center = plate2_position[0]

    #     Ex_p1 = 0.0
    #     Ex_p2 = 0.0
    #     Ey_p1 = 0.0
    #     Ey_p2 = 0.0
    #     Ez_p1 = 0.0
    #     Ez_p2 = 0.0
    
    #     if (plate_height != 0.0) and (plate_width != 0.0):    
    #         Ex_p1, Ey_p1, Ez_p1 = plate_field(x, y, z, plate1_center, V_plate1)
    #         Ex_p2, Ey_p2, Ez_p2 = plate_field(x, y, z, plate2_center, V_plate2)
    
    #     # --- Sum all contributions ---
    #     return np.array([
    #         Ex + Ex_rod + Ex_p1 + Ex_p2,
    #         Ey + Ey_rod + Ey_p1 + Ey_p2,
    #         Ez + Ez_rod + Ez_p1 + Ez_p2
    #     ])
    
    

    V_nozzle = params.param1
    V_rod = params.param2
    rod_z = params.param3
    rod_diameter = params.param4
    rod_length = params.param5
    distance_nozzle_rod = params.param6
    n_nozzles = int(params.param7)
    nozzles_spacing = params.param8
    nozzles_center = params.param9
    spacing = nozzles_spacing  # distance between nozzles along x-axis
    nozzle_z = rod_z + distance_nozzle_rod
    z_slice = params.param10
    nozzles_shift = params.param11
    V_shields = params.param12
    
    plate_height = params.param13      # along y-axis
    plate_width = params.param14       # along z-axis
    plate_spacing = params.param15
    
    x_slice = params.param16
    y_slice = params.param17
    
    V_plate1 = V_shields
    V_plate2 = V_shields
    
    nozzle_length = plate_height
    
    nozzle_positions = [(x0-((n_nozzles-1)*spacing)/2+nozzles_center, nozzles_shift, nozzle_z) for x0 in 
                     np.linspace(0,
                                 (n_nozzles-1)*spacing,
                                 n_nozzles)]
    plate1_position = [(0.0-((n_nozzles-1)*spacing)/2+nozzles_center-plate_spacing, nozzles_shift, nozzle_z+plate_height/2.0 - 1.0)]
    plate2_position = [((n_nozzles-1)*spacing-((n_nozzles-1)*spacing)/2+nozzles_center+plate_spacing, nozzles_shift, nozzle_z+plate_height/2.0 - 1.0)]
    
    
    # anything above threshold is shown as yellow
    #threshold = 2  # adjust based on your units
    threshold = params.param18
    slice_choice = int(params.param19)
    
#    print("slice_choice = ", slice_choice)
    
    class ThresholdNorm(mcolors.Normalize):
        def __init__(self, vmin=None, vmax=None, threshold=None, clip=False):
            super().__init__(vmin, vmax, clip)
            self.threshold = threshold
    
        def __call__(self, value, clip=None):
            # Normalize as usual
            res = super().__call__(value, clip)
            # Force values above threshold to 1.0 (top of colormap)
            if self.threshold is not None:
                res = np.ma.masked_array(res, mask=np.isnan(value))
                res[value >= self.threshold] = 1.0
            return res
        
    
    
    # Electric field (nozzle + rod)
    def electric_field(x, y, z):
        
        # Start with zero field
        Ex, Ey, Ez = 0.0, 0.0, 0.0
        
        # Contributions from each nozzle (point charge approximation)
        for (xn, yn, zn) in nozzle_positions:
            rx, ry, rz = (x - xn), (y - yn), (z - zn)
            r3 = (rx**2 + ry**2 + rz**2 + 1e-9)**1.5
            Ex += V_nozzle * rx / r3
            Ey += V_nozzle * ry / r3
            Ez += V_nozzle * rz / r3
        
        # Contribution from the rod (cylinder along x-axis)
#        N_seg = 50
        N_seg = 50
        xs = np.linspace(-rod_length/2, rod_length/2, N_seg)
        Ex_rod = Ey_rod = Ez_rod = 0.0
        for xi in xs:
            r_seg = np.sqrt(((x - xi))**2 + (y)**2 + ((z - rod_z))**2)
            Ex_rod += V_rod * ((x - xi)) / (r_seg**3 + 1e-9)
            Ey_rod += V_rod * (y) / (r_seg**3 + 1e-9)
            Ez_rod += V_rod * ((z - rod_z)) / (r_seg**3 + 1e-9)
            
        def plate_field(x, y, z, plate_center, V_plate):
            # plate_center = (xp, yp, zp)
            xp, yp, zp = plate_center
            Ny, Nz = 10, 10  # resolution of discretization
            ys = np.linspace(yp - plate_height/2, yp + plate_height/2, Ny)
            zs = np.linspace(zp - plate_width/2, zp + plate_width/2, Nz)
            Ex_p, Ey_p, Ez_p = 0.0, 0.0, 0.0
            for yi in ys:
                for zi in zs:
                    rx, ry, rz = (x - xp), (y - yi), (z - zi)
                    r3 = (rx**2 + ry**2 + rz**2 + 1e-9)**1.5
                    Ex_p += V_plate * rx / r3
                    Ey_p += V_plate * ry / r3
                    Ez_p += V_plate * rz / r3
            return Ex_p, Ey_p, Ez_p
       
        # X-positions of the plates
        # ------------------------------
       
        plate1_center = plate1_position[0]
        plate2_center = plate2_position[0]
    
        Ex_p1 = 0.0
        Ex_p2 = 0.0
        Ey_p1 = 0.0
        Ey_p2 = 0.0
        Ez_p1 = 0.0
        Ez_p2 = 0.0
    
        if (plate_height != 0.0) and (plate_width != 0.0):
            Ex_p1, Ey_p1, Ez_p1 = plate_field(x, y, z, plate1_center, V_plate1)
            Ex_p2, Ey_p2, Ez_p2 = plate_field(x, y, z, plate2_center, V_plate2)
    
    
        return np.array([Ex + Ex_rod + Ex_p1 + Ex_p2,
                         Ey + Ey_rod + Ey_p1 + Ey_p2,
                         Ez + Ez_rod + Ez_p1 + Ez_p2])
    
    
    
    # --- 2D slice with field strength + field lines in x–z plane (y=0) ---

    if slice_choice == 0:  # x-z
    
        # Define grid
        nx, nz = 200, 200  # resolution
        y0 = y_slice
        
        x_vals = np.linspace(-rod_length/2-2.0, rod_length/2+2.0, nx)
        z_vals = np.linspace(-2,
                             nozzle_z + 0.5*distance_nozzle_rod, nz)
        
        
        X, Z = np.meshgrid(x_vals, z_vals)
        #Y = np.zeros_like(X)  # y=0 plane
        Y = np.full_like(X, y0)  # fixed x-slice
        
        # Compute field strength and field vectors
        E_slice = np.zeros_like(X)
        Ex_slice = np.zeros_like(X)
        Ez_slice = np.zeros_like(X)
        
        for i in range(nx):
            for j in range(nz):
                E = electric_field(X[j, i], y0, Z[j, i])  # y=0 plane
                Ex_slice[j, i] = E[0]
                Ez_slice[j, i] = E[2]
                E_slice[j, i] = np.sqrt(E[0]**2 + E[2]**2)


        # --- Prepare 2D interpolators for x–z slice ---
        # Ex_slice and Ez_slice have shape (nz, nx) with coords (z_vals, x_vals)
        interp_Ex_xz = RegularGridInterpolator((z_vals, x_vals), Ex_slice,
                                               bounds_error=False, fill_value=0.0)
        interp_Ez_xz = RegularGridInterpolator((z_vals, x_vals), Ez_slice,
                                               bounds_error=False, fill_value=0.0)
        
        # Seeds (same as you had)
        Nseeds_per_nozzle = 180
        seed_radius = 0.2
        seeds = []
        for (xn, yn, zn) in nozzle_positions:
            angles = np.linspace(0, 2*np.pi, Nseeds_per_nozzle, endpoint=False)
            for a in angles:
                xs = xn + seed_radius * np.cos(a)
                zs = zn + seed_radius * np.sin(a)
                seeds.append((xs, zs))
        
        # Integrate streamlines using interpolators
        hits = []
        path_lengths = []
        total = 0
        max_steps = 2000
        ds = 0.1   # step length (tune to your units)
        
        # Parameters for jet spreading
        k_sigma = 0.02     # scaling factor (tune!)
        alpha = 0.5        # exponent (0.5 ~ sqrt law, 1.0 = linear)
                
        x_min, x_max = x_vals[0], x_vals[-1]
        z_min, z_max = z_vals[0], z_vals[-1]
        
        for (xs, zs) in seeds:
            x, z = float(xs), float(zs)
            path_length = 0.0 
            for _ in range(max_steps):
                Ex = float(interp_Ex_xz((z, x)))   # note order (z, x)
                Ez = float(interp_Ez_xz((z, x)))
                norm1 = np.hypot(Ex, Ez)
                if norm1 < 1e-12:
                    # field too small → consider this streamline escaping
                    break
                dx = (Ex / norm1) * ds
                dz = (Ez / norm1) * ds
                x += dx
                z += dz
                path_length += ds  # accumulate travel length
        
                # If outside the plotting domain, stop (escaped)
                if (x < x_min - 1.0) or (x > x_max + 1.0) or (z < z_min - 1.0) or (z > z_max + 1.0):
                    break
        
                # Rod hit check (x–z slice): z close to rod_z and x within rod length
                if (abs(z - rod_z) <= rod_diameter/2) and (-rod_length/2.0 <= x <= rod_length/2.0):
                    hits.append((x, z))
                    path_lengths.append(path_length)
                    break
            total += 1
        
        efficiency = len(hits) / max(1, total)
        #print(f"[Metrics] x–z slice capture efficiency: {efficiency:.2f}")
        
        if False: #hits:
            hit_xs = [x for (x, z) in hits]
            hist, bins = np.histogram(hit_xs, bins=60, range=(-rod_length/2.0, rod_length/2.0))
            bin_width = bins[1] - bins[0]
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            
            # Raw histogram (counts, not normalized yet)
            hist, bins = np.histogram(hit_xs, bins=24, range=(-rod_length/2.0, rod_length/2.0))
            bin_width = bins[1] - bins[0]
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            
            # --- Apply Gaussian smoothing on raw counts ---
            sigma_bins = 2.0   # in *number of bins*, not physical cm
            hist_smooth_counts = gaussian_filter1d(hist.astype(float), sigma=sigma_bins, mode="constant")
            
            # Normalize smoothed counts to a density (so integral = 1)
            if hist_smooth_counts.sum() > 0:
                hist_smooth_density = hist_smooth_counts / (hist_smooth_counts.sum() * bin_width)
            else:
                hist_smooth_density = np.zeros_like(hist_smooth_counts)

            
            #hist_density = hist / (hist.sum() * bin_width)  # normalized per unit length
           # print(f"[Metrics] Hit density histogram (per unit length): {hist_density.round(3).tolist()}")
            #hist, bins = np.histogram(hit_xs, bins=12, range=(-rod_length/2.0, rod_length/2.0))
            #print(f"[Metrics] Hit density histogram (rod length): {hist.tolist()}")
            # optionally also print a few raw x hits for debugging:
            # print("raw hit x positions (first 20):", np.array(hit_xs)[:20])  
            
        if hits:
            # Bin centers
            n_bins = 100
            bin_edges = np.linspace(-rod_length/2.0, rod_length/2.0, n_bins+1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            deposition = np.zeros_like(bin_centers)
        
            # Add Gaussian from each hit
            for x_hit, L in zip(hits, path_lengths):
                sigma = k_sigma * (L ** alpha)
                deposition += pdf(bin_centers, loc=x_hit, scale=sigma)
        
            # Normalize
            deposition /= deposition.sum()
            
                
        # Plot heatmap of field strength
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        threshold = params.param18
        
#        threshold = 10  # in your units, e.g. V/m or kV/cm depending on inputs
        cmap = plt.cm.plasma
        norm1 = ThresholdNorm(vmin=0, vmax=threshold, threshold=threshold)
        
        im = ax2.pcolormesh(X, Z, E_slice, cmap=cmap, norm=norm1, shading="auto")
        fig2.colorbar(im, ax=ax2, orientation="horizontal", shrink=0.8, label="|E|")
        
        # Add 2D streamlines (direction field)
        ax2.streamplot(X, Z, Ex_slice, Ez_slice, color="white",
                       linewidth=0.7, density=1.2, arrowsize=0.6)
        
        # Add rod projection (circle at z=rod_z)
        rod_rect = patches.Rectangle(
            (-rod_length/2, rod_z - rod_diameter/2),  # (x, z) lower-left corner
            rod_length,                               # width (x direction)
            rod_diameter,                             # height (z direction)
            color="grey", alpha=0.6, zorder=10
        )
        ax2.add_patch(rod_rect)
        
        
        # Add nozzle positions (as red dots)
        for (xn, yn, zn) in nozzle_positions:
            ax2.plot(xn, zn, "ro")
            
        # Add plate projections (vertical lines)
        if (plate_height != 0.0) and (plate_width != 0.0):
            for plate_center, color in zip([plate1_position[0], plate2_position[0]], ["blue", "green"]):
                xp, yp, zp = plate_center
                ax2.add_line(plt.Line2D([xp, xp], [zp - plate_width/2, zp + plate_width/2],
                                        color=color, linewidth=2, alpha=0.6))
        if hits:
            # Add second y-axis for histogram overlay
            ax_hist = ax2.twinx()
            #hist_smooth = gaussian_filter1d(hist_density, sigma=2)
#            ax_hist.plot(bin_centers, hist_smooth_density, color="black", linewidth=2, label="Hit density")
            ax_hist.plot(bin_centers, deposition, color="black", linewidth=2, label="Hit density")
            ax_hist.set_ylabel("Hit density (fraction)", color="black")
            ax_hist.set_ylim(0, np.max(hist_smooth_density) *1.2 )  # Max bar = 50% of plot height
            ax_hist.tick_params(axis="y", labelcolor="black")
            # Add text at specific coordinates
            ax_hist.text(
                0.02, 0.95, 
                f"Field efficiency: {efficiency:.2f}", 
                transform=ax_hist.transAxes,   # <--- important
                fontsize=10, color="white", 
                ha="left", va="top"
            )
        
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        ax2.set_title(f"2D field strength and field lines (y={y0:.1f} plane)")
        plt.tight_layout()        
        buf = io.BytesIO()
        fig2.savefig(buf, format="png")   # safer than plt.savefig
        buf.seek(0)
        plt.close(fig2)
        
#        print("Returning image, size:", buf.getbuffer().nbytes)

        return StreamingResponse(buf, media_type="image/png")
     
    
    # --- 2D slice with field strength + field lines in y–z plane (x=0) ---

    if slice_choice == 1:  # y-z
    
        ny, nz = 200, 200  # resolution
        x0= x_slice
        y_vals = np.linspace(-20, 20, ny)
        z_vals = np.linspace(-2,
                             nozzle_z + 0.5*distance_nozzle_rod, nz)
        
        Y, Z = np.meshgrid(y_vals, z_vals)
        #X = np.zeros_like(Y)  # x=0 plane
        X = np.full_like(Y, x0)  # fixed x-slice
        
        # Compute field strength and field vectors
        E_slice = np.zeros_like(Y)
        Ey_slice = np.zeros_like(Y)
        Ez_slice = np.zeros_like(Y)
        
        for i in range(ny):
            for j in range(nz):
                E = electric_field(x0, Y[j, i], Z[j, i])  # x=0 plane
                Ey_slice[j, i] = E[1]
                Ez_slice[j, i] = E[2]
                E_slice[j, i] = np.sqrt(E[1]**2 + E[2]**2)
                
        # --- Prepare 2D interpolators for y–z slice ---
        # Ey_slice and Ez_slice have shape (nz, ny) with coords (z_vals, y_vals)
        interp_Ey_yz = RegularGridInterpolator((z_vals, y_vals), Ey_slice,
                                               bounds_error=False, fill_value=0.0)
        interp_Ez_yz = RegularGridInterpolator((z_vals, y_vals), Ez_slice,
                                               bounds_error=False, fill_value=0.0)
        
        # Seeds on circle around each nozzle (projected to y–z)
        Nseeds_per_nozzle = 180
        seed_radius = 0.2
        seeds = []
        for (xn, yn, zn) in nozzle_positions:
            angles = np.linspace(0, 2*np.pi, Nseeds_per_nozzle, endpoint=False)
            for a in angles:
                ys = yn + seed_radius * np.cos(a)
                zs = zn + seed_radius * np.sin(a)
                seeds.append((ys, zs))
        
        hits = []
        total = 0
        max_steps = 2000
        ds = 0.1
        
        y_min, y_max = y_vals[0], y_vals[-1]
        z_min, z_max = z_vals[0], z_vals[-1]
        
        for (ys, zs) in seeds:
            y, z = float(ys), float(zs)
            for _ in range(max_steps):
                Ey = float(interp_Ey_yz((z, y)))   # order (z, y)
                Ez = float(interp_Ez_yz((z, y)))
                norm = np.hypot(Ey, Ez)
                if norm < 1e-12:
                    break
                dy = (Ey / norm) * ds
                dz = (Ez / norm) * ds
                y += dy
                z += dz
        
                if (y < y_min - 1.0) or (y > y_max + 1.0) or (z < z_min - 1.0) or (z > z_max + 1.0):
                    break
        
                # Rod hit check (y–z slice): distance to rod axis
                if (y**2 + (z - rod_z)**2) <= (rod_diameter/2)**2:
                    hits.append((y, z))
                    break
            total += 1
        
        efficiency = len(hits) / max(1, total)
        print(f"[Metrics] y–z slice capture efficiency: {efficiency:.2f}")
        
        # if hits:
        #     hit_angles = [np.arctan2(y, z - rod_z) for (y, z) in hits]
        #     hist, bins = np.histogram(hit_angles, bins=12, range=(-np.pi, np.pi))
        #     print(f"[Metrics] Hit density histogram (angles): {hist.tolist()}")
        #     print("raw hit angles (radians) first 20:", np.array(hit_angles)[:20])  
        
              
        # Plot heatmap of field strength
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        
#        threshold = 10  # adjust units as before
        threshold = params.param18
        cmap = plt.cm.plasma
        norm = ThresholdNorm(vmin=0, vmax=threshold, threshold=threshold)
        
        im = ax3.pcolormesh(Y, Z, E_slice, cmap=cmap, norm=norm, shading="auto")
        fig3.colorbar(im, ax=ax3, shrink=0.8, label="|E|")
        
        # Add streamlines
        ax3.streamplot(Y, Z, Ey_slice, Ez_slice, color="white",
                       linewidth=0.7, density=1.2, arrowsize=0.6)
        
        # Add rod projection (circle in y–z plane at x=0)
        rod_circle = patches.Circle(
            (0, rod_z),            # center (y=0, z=rod_z)
            radius=rod_diameter/2, # radius
            color="grey", alpha=0.6, zorder=10
        )
        ax3.add_patch(rod_circle)
        
        # Add nozzle positions (as red dots projected in y–z plane)
        for (xn, yn, zn) in nozzle_positions:
            ax3.plot(yn, zn, "ro")  # x is 0 in this slice → project onto y–z
            
        # Add plate projections (rectangles in y–z plane at x=0)
        if (plate_height != 0.0) and (plate_width != 0.0):
            for plate_center, color in zip([plate1_position[0], plate2_position[0]], ["blue", "green"]):
                xp, yp, zp = plate_center
                rect = patches.Rectangle(
                    (yp - plate_height/2, zp - plate_width/2),
                    plate_height, plate_width,
                    fill=True, color=color, alpha=0.3, zorder=5
                )
                ax3.add_patch(rect)

        if hits:
            ax3.text(
                0.02, 0.95, 
                f"Field efficiency: {efficiency:.2f}", 
                transform=ax3.transAxes,   # <--- important
                fontsize=10, color="white", 
                ha="left", va="top"
            )
        ax3.set_xlabel("y")
        ax3.set_ylabel("z")
        ax3.set_aspect("equal")
        ax3.set_title(f"2D field strength and field lines (x={x0:.1f} plane)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig3)

        return StreamingResponse(buf, media_type="image/png")

    
    # --- 2D slice with field strength in x–y plane (z = z_slice) ---
    
    if slice_choice == 2:  # x-y


        nx, ny = 200, 200
        
        z0 = z_slice
        
        x_vals = np.linspace(-rod_length/2 - 2.0, rod_length/2 + 2.0, nx)
        y_vals = np.linspace(-rod_length/2 - 2.0, rod_length/2 + 2.0, ny)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.full_like(X, z0)  # fixed z-slice
        
        # Compute field strength and in-plane vectors
        E_slice = np.zeros_like(X)
        Ex_slice = np.zeros_like(X)
        Ey_slice = np.zeros_like(X)
        
        for i in range(nx):
            for j in range(ny):
                E = electric_field(X[j, i], Y[j, i], z0)  # evaluate at z=z0
                Ex_slice[j, i] = E[0]
                Ey_slice[j, i] = E[1]
                E_slice[j, i] = np.sqrt(E[0]**2 + E[1]**2 + E[2]**2)
        
        # Plot heatmap of field strength
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        
#        threshold = 5
        threshold = params.param18
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=0, vmax=threshold)
        
        im = ax4.pcolormesh(X, Y, E_slice, cmap=cmap, norm=norm, shading="auto")
        fig4.colorbar(im, ax=ax4, shrink=0.8, label="|E| (kV/cm)")
        
        # Add 2D streamlines in (x, y) plane
        ax4.streamplot(X, Y, Ex_slice, Ey_slice, color="white",
                      linewidth=0.7, density=1.2, arrowsize=0.6)
        
        
        # Mark nozzle projections (only those close to z0 will appear meaningful)
        for (xn, yn, zn) in nozzle_positions:
            ax4.plot(xn, yn, "ro")
        
        
        # Draw rod cross-section (circle at (0,0), z=z0 ≈ rod_z)
        rect_top = plt.Rectangle((-rod_length/2, -rod_diameter/2),
                                 rod_length, rod_diameter,
                                 fill=True, color='grey')
        ax4.add_patch(rect_top)
        
        # Add plate projections in x–y slice (rectangles spanning y)
        if (plate_height != 0.0) and (plate_width != 0.0):
            for plate_center, color in zip([plate1_position[0], plate2_position[0]], ["blue", "green"]):
                xp, yp, zp = plate_center
            #    if abs(z0 - zp) < plate_width/2:  # only visible if slice intersects plat 
                rect = plt.Rectangle((xp-0.05, yp - plate_height/2),
                                         0.1, plate_height,
                                         fill=True, color=color, alpha=0.3, zorder=5)
                ax4.add_patch(rect)
        
        
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        ax4.set_title(f"2D field slice (x–y plane, z={z0:.1f})")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig4)

        return StreamingResponse(buf, media_type="image/png")
       
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
