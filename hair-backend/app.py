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
import cv2
import numpy as np

app = FastAPI()

# ✅ Load model ONCE at startup
hair_segmenter_base_options = python.BaseOptions(model_asset_path="hair_segmenter.tflite")
hair_segmenter_options = vision.ImageSegmenterOptions(
    base_options=hair_segmenter_base_options,
    output_confidence_masks=True
)
hair_segmenter = vision.ImageSegmenter.create_from_options(hair_segmenter_options)



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


@app.post("/")
async def segment_hair(file: UploadFile = File(...)):
    
    # Read uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Run segmentation
    hair_mask = detect_hair_with_segmenter(image, hair_segmenter)
    # Flip the binary mask. Needed because of a bug in the app.
    flipped_hair_mask = np.flipud(hair_mask)

    # Assuming binary_hair_mask is a uint8 mask with 0 = background, 255 = hair
    # Convert to .png and return
    _, png_data = cv2.imencode(".png", flipped_hair_mask)

    return Response(content=png_data.tobytes(), media_type="image/png")
