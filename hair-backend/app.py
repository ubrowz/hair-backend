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
    N_fibers = spinning_time * 60
   
    random.seed()

    rect_length = 150 * math.sqrt(canvas_width**2 + canvas_height**2)

    noz_angle = math.degrees(math.atan(nozzle_speed /(math.pi * diameter * (rot_speed /60))))

    director_one = director + noz_angle
    director_two = director - noz_angle
    
    currentAxis = plt.gca()
    
    fibers_in_layer = N_fibers // layers

    for i in range(N_fibers):
        
        fiber_range = i // fibers_in_layer
        fiber_color = 0.028+(0.37/layers)*fiber_range+(0.37/layers)*random.random()
        fiber_depth = layers - fiber_range
        fiber_delta_width = fiber_depth * 0.015 * width_basic
        
        
        if i % 2 == 0:
            randangle = rand_angle(director_one ,sigma_angle)
            randcoord = rand_coord(randangle)
            vierhoek = Rectangle(randcoord ,width_basic - fiber_delta_width ,rect_length ,angle=randangle ,lw=1 ,ec="white"
                                  ,fc=str(fiber_color))

            currentAxis.add_patch(vierhoek)
        else:
            randangle = rand_angle(director_two ,sigma_angle)
            randcoord = rand_coord(randangle)
            vierhoek = Rectangle(randcoord ,width_basic - fiber_delta_width ,rect_length ,angle=randangle ,lw=1 ,ec="white"
                                  ,fc=str(fiber_color))

            currentAxis.add_patch(vierhoek)

    currentAxis.set_aspect(aspect=1)            
    plt.xlim([0 ,canvas_width])
    plt.ylim([0 ,canvas_height])

    plt.axis("off")
    currentAxis.add_patch(plt.Rectangle((0 ,0), 1, 1, facecolor=(0 ,0 ,0) ,transform=currentAxis.transAxes, zorder=-1))
    
    buf = io.Bytes.IO()
    
    plt.savefig(buf, format="png")
    buf.seek(0)

    plt.close()

    return StreamingResponse(buf, media_type="image/png")
