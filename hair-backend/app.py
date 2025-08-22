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
from fastapi.responses import FileResponse
from fastapi.responses import Response
import cv2
import numpy as np
import tempfile

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
#    print(hair_mask)
    
    # Use a higher threshold for more precise hair detection
    binary_hair_mask = (1-(hair_mask > 0.7).astype(np.uint8)) * 255  # Increased from 0.5 to 0.7
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    binary_hair_mask = cv2.morphologyEx(binary_hair_mask, cv2.MORPH_CLOSE, kernel)
    binary_hair_mask = cv2.morphologyEx(binary_hair_mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate hair coverage percentage
    total_pixels = binary_hair_mask.shape[0] * binary_hair_mask.shape[1]
    hair_pixels_count = np.sum(binary_hair_mask > 0)
    hair_coverage = hair_pixels_count / total_pixels
    
    # Check if significant hair is detected (at least 3% of image)
    if hair_coverage < 0.03:  # Reduced from 0.05 to 0.03
        return binary_hair_mask
    
    # Extract hair pixels for color analysis with additional filtering
    hair_pixel_mask = (binary_hair_mask > 0)
    hair_pixels = rgb_image[hair_pixel_mask]
    
    if len(hair_pixels) > 10:  # Need minimum pixels for reliable analysis
        
        # Additional filtering: Remove outlier colors that might be background/clothing
        # Calculate median color to be more robust against outliers
        median_color = np.median(hair_pixels, axis=0)
        
        # Calculate distances from median color
        color_distances = np.linalg.norm(hair_pixels - median_color, axis=1)
        
        # Keep only pixels within reasonable distance from median (remove outliers)
        distance_threshold = np.percentile(color_distances, 85)  # Keep 85% closest to median
        valid_pixels = hair_pixels[color_distances <= distance_threshold]
        
        if len(valid_pixels) > 5:
            # Use median instead of mean for more robust color calculation
            avg_hair_color_rgb = np.median(valid_pixels, axis=0)
            avg_hair_color_bgr = avg_hair_color_rgb[::-1]
            
            # Additional sanity check: make sure it's not skin-colored
            # Skin colors typically have: R > G > B and are relatively bright
            r, g, b = avg_hair_color_rgb
            brightness = np.mean(avg_hair_color_rgb)
            
            # If it looks like skin color, try using the darkest 30% of pixels instead
            if r > g > b and brightness > 120:  # Likely skin-colored
                print("Detected potential skin contamination, using darker pixels...")
                pixel_brightness = np.mean(valid_pixels, axis=1)
                dark_threshold = np.percentile(pixel_brightness, 30)
                dark_pixels = valid_pixels[pixel_brightness <= dark_threshold]
                
                if len(dark_pixels) > 5:
                    avg_hair_color_rgb = np.median(dark_pixels, axis=0)
                    avg_hair_color_bgr = avg_hair_color_rgb[::-1]
                    
            
            return binary_hair_mask
    
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


    # Invert the mask
    # inverted_mask = cv2.bitwise_not(hair_mask)
    # _, png_data = cv2.imencode(".png", inverted_mask)

    # return Response(content=png_data.tobytes(), media_type="image/png")

    # Save inverted mask
#    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#    cv2.imwrite(tmpfile.name, inverted_mask)
    

    # Save mask temporarily
#    cv2.imwrite(tmpfile.name, hair_mask)

#    return FileResponse(tmpfile.name, media_type="image/png")

