import os
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import shutil
import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import imageio
import processing_utils as pc
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.measure import label, regionprops
from skimage import io, color, filters, morphology, measure
from skimage.transform import rescale
from skimage.filters import threshold_otsu
from skimage.filters import frangi
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter,binary_dilation
from collections import Counter
from skimage.morphology import remove_small_objects, binary_closing
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def folder_contains_rca_mpr_or_curved_updated(file_name, file_path):
    # Check if the file starts with "SC"
    if file_name.startswith("SC"):
        # Read the DICOM file
        dicom_info = pydicom.dcmread(file_path)

        # Check if "LAD", "MPR", or "Curved" are in Series Description
        if hasattr(dicom_info, 'SeriesDescription') and dicom_info.SeriesDescription  and ("RCA" in dicom_info.SeriesDescription or "RCA_2" in dicom_info.SeriesDescription or "RCA_3" in dicom_info.SeriesDescription) and ("MPR" in dicom_info.SeriesDescription or "Curved" in dicom_info.SeriesDescription):
            # Return True if LAD MPR or Curved DICOM files are found
            return True

    # Return False if no LAD MPR or Curved DICOM files are found
    return False

def folder_contains_lad_mpr_or_curved_updated(file_name, file_path):
    # Check if the file starts with "SC"
    if file_name.startswith("SC"):
        # Read the DICOM file
        dicom_info = pydicom.dcmread(file_path)

        # Check if "LAD", "MPR", or "Curved" are in Series Description
        if hasattr(dicom_info, 'SeriesDescription') and dicom_info.SeriesDescription  and ("LAD" in dicom_info.SeriesDescription or "LAD_2" in dicom_info.SeriesDescription or "LAD_8" in dicom_info.SeriesDescription) and ("MPR" in dicom_info.SeriesDescription or "Curved" in dicom_info.SeriesDescription):
            # Return True if LAD MPR or Curved DICOM files are found
            return True

    # Return False if no LAD MPR or Curved DICOM files are found
    return False

def folder_contains_cx_mpr_or_curved_updated(file_name, file_path):
    # Check if the file starts with "SC"
    if file_name.startswith("SC"):
        # Read the DICOM file
        dicom_info = pydicom.dcmread(file_path)

        # Check if "LAD", "MPR", or "Curved" are in Series Description
        if hasattr(dicom_info, 'SeriesDescription') and dicom_info.SeriesDescription  and ("CX" in dicom_info.SeriesDescription or "CX_2" in dicom_info.SeriesDescription  or "LCX" in dicom_info.SeriesDescription or "CX-OM" in dicom_info.SeriesDescription or "CX_PDA" in dicom_info.SeriesDescription or "CX_OM1" in dicom_info.SeriesDescription) and ("MPR" in dicom_info.SeriesDescription or "Curved" in dicom_info.SeriesDescription):
            # Return True if LAD MPR or Curved DICOM files are found
            return True

    # Return False if no LAD MPR or Curved DICOM files are found
    return False

def fetch_artery_with_bounday_all_slices_3ch_new(patient_ids, dicom_ids,dicom_fold_num, destination_dir, artery_to_process, file_extension='_rgb.png', window_min = 130, window_max = 600,block_size=1, threshold_value=.125,min_size = 20,border_pixels = 1000000, border_pixels_2 = 20):
    for patient_id, dicom_id,dicom_fold_num in zip(patient_ids, dicom_ids,dicom_fold_num):
        source_path = os.path.join('E:/Dicom', patient_id, dicom_id)
        if os.path.exists(source_path):
            destination_path = os.path.join(destination_dir, f"{patient_id}_{dicom_fold_num}")
            os.makedirs(destination_path, exist_ok=True)
            for folder_name, subfolders, files in os.walk(source_path):
                for file_name in files:
                    file_path = os.path.join(folder_name, file_name)
                    # Check if the folder contains MPR or Curved files
                    if artery_to_process=='LAD':
                    
                        if folder_contains_lad_mpr_or_curved_updated(file_name, file_path):
                            #print(file_path)
                            dicom_slice = imageio.imread(file_path)
                            #dicom_slice = np.clip(dicom_slice, -1000, 2000)
                            #artery,mapped_image_cleaned, dicom_slice=pc.windowing_based_HU(dicom_slice, window_min = window_min, window_max = window_max,block_size=block_size, threshold_value=threshold_value,min_size =min_size,border_pixels = border_pixels, border_pixels_2=border_pixels_2)
                            artery=windowing_sobel_HU(dicom_slice, file_path,window_min = window_min, window_max = window_max)
                            output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                            # Scale pixel values to the range [0, 255]
                            artery_rgb = np.stack((artery,) * 3, axis=-1)
                            artery_rgb = (artery_rgb - artery_rgb.min()) / (artery_rgb.max() - artery_rgb.min()) * 255
                            artery_rgb = artery_rgb.astype('uint8')
                            output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                            # Save the image
                            imageio.imwrite(output_file, artery_rgb)

                    if artery_to_process=='RCA':
                    
                        if folder_contains_rca_mpr_or_curved_updated(file_name, file_path):
                            dicom_slice = imageio.imread(file_path)
                            dicom_slice = np.clip(dicom_slice, -1000, 2000)
                            #artery,mapped_image_cleaned, dicom_slice=pc.windowing_based_HU(dicom_slice, window_min = window_min, window_max = window_max,block_size=block_size, threshold_value=threshold_value,min_size =min_size,border_pixels = border_pixels, border_pixels_2=border_pixels_2)
                            artery=windowing_sobel_HU(dicom_slice, window_min = window_min, window_max = window_max)
                            output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                            # Scale pixel values to the range [0, 255]
                            artery_rgb = np.stack((artery,) * 3, axis=-1)
                            artery_rgb = (artery_rgb - artery_rgb.min()) / (artery_rgb.max() - artery_rgb.min()) * 255
                            artery_rgb = artery_rgb.astype('uint8')
                            output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                            # Save the image
                            imageio.imwrite(output_file, artery_rgb) 

                    if artery_to_process=='CX':
                    
                        if folder_contains_cx_mpr_or_curved_updated(file_name, file_path):
                            dicom_slice = imageio.imread(file_path)
                            dicom_slice = np.clip(dicom_slice, -1000, 2000)
                            #artery,mapped_image_cleaned, dicom_slice=pc.windowing_based_HU(dicom_slice, window_min = window_min, window_max = window_max,block_size=block_size, threshold_value=threshold_value,min_size =min_size,border_pixels = border_pixels, border_pixels_2=border_pixels_2)
                            artery=windowing_sobel_HU(dicom_slice, window_min = window_min, window_max = window_max)
                            output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                            # Scale pixel values to the range [0, 255]
                            artery_rgb = np.stack((artery,) * 3, axis=-1)
                            artery_rgb = (artery_rgb - artery_rgb.min()) / (artery_rgb.max() - artery_rgb.min()) * 255
                            artery_rgb = artery_rgb.astype('uint8')
                            output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                            # Save the image
                            imageio.imwrite(output_file, artery_rgb)                

    return  artery


def clip_and_normalize(image, min_val, max_val):
    clipped = np.clip(image, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Ensure the image is in 8-bit format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255)  # Ensure values are within byte range
        image = (image * 255).astype(np.uint8)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    clahe_image = clahe.apply(image)
    return clahe_image



def fill_bounding_box(image, bounding_boxes, window_size=10):
    image_filled = image.copy()
    
    for (x, y, w, h) in bounding_boxes:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Extract the surrounding region
        surrounding_region = image_filled[max(y1 - window_size, 0):min(y2 + window_size, image_filled.shape[0]),
                                          max(x1 - window_size, 0):min(x2 + window_size, image_filled.shape[1])]
        
        # Calculate average value of the surrounding pixels
        avg_pixel_value = np.median(surrounding_region)
        
        # Fill the bounding box area with the average value
        image_filled[y1-5:y2+5, x1-10:x2+10] = avg_pixel_value
    
    return image_filled

def clean_text(text):
    # Remove unwanted special characters and formatting
    text = re.sub(r'[{}\\]', '', text)  # Remove curly braces and backslashes
    text = re.sub(r'\s*\(\d*\)\s*', '', text)  # Remove patterns like '(1)' with optional surrounding spaces
    text = re.sub(r"'", '', text)  # Remove single quotes
    text = re.sub(r'[^a-zA-Z0-9_ ]', '', text)  # Keep only alphanumeric, underscores, and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text


def windowing_sobel_HU(dicom_slice, file_path,window_min = -55, window_max = 400):

    dicom_slice1 = np.clip(dicom_slice, -1000, 2000)

    ranges = [(-150, -50), (-50, 50), (50, 130), (130, 400), (400, 1000)]
    weights = [0.3, 0.2, 0.1,0.1,.3]

    accumulated_edges = np.zeros_like(dicom_slice1, dtype=float)
    combined_normalized_image = np.zeros_like(dicom_slice1, dtype=float)

    for i, (min_val, max_val) in enumerate(ranges):
        normalized_image = clip_and_normalize(dicom_slice1, min_val, max_val)
        edges = filters.sobel(normalized_image)
        accumulated_edges = np.maximum(accumulated_edges, edges)
        combined_normalized_image += weights[i] * normalized_image

    combined_edges = np.clip(accumulated_edges, 0, 1)
    binary_image1 = combined_edges > .35
    binary_image1 = remove_small_objects(binary_image1, min_size=25)
    binary_image1=binary_image1*combined_edges
    combined_normalized_image = apply_clahe(combined_normalized_image)
    #enhanced_edges = np.clip(combined_normalized_image+ 4*binary_image1, 0, 1) 
    enhanced_edges = combined_normalized_image + 4*binary_image1
    thresh = threshold_otsu(enhanced_edges )
    binary_image = enhanced_edges  > thresh

    labeled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)
    regions = measure.regionprops(labeled_image)
    top_by_area = sorted(regions, key=lambda r: r.area, reverse=True)[:1]  # Top 5 by area
    top_by_major_axis = sorted(regions, key=lambda r: r.major_axis_length, reverse=True)[:1]  # Top 5 by major axis length
    top_by_perimeter = sorted(regions, key=lambda r: r.perimeter, reverse=True)[:1]  # Top 5 by perimeter
    top_regions_centroids = [
    region.centroid for region in top_by_area + top_by_major_axis + top_by_perimeter
    ]

    # Count occurrences of each centroid
    region_counter = Counter(top_regions_centroids)
    most_common_centroids = region_counter.most_common()
    selected_region = None
    if most_common_centroids:
        most_common_centroid = most_common_centroids[0][0]
        for region in regions:
            if region.centroid == most_common_centroid:
                selected_region = region
                break

    selected_region_image = np.zeros_like(binary_image, dtype=np.uint8)
    if selected_region:
        for coords in selected_region.coords:
            selected_region_image[coords[0], coords[1]] = 1
    

    border_pixels = 20 
    rows, cols = selected_region_image.shape
    bordered_arr = np.zeros((rows, cols), dtype=selected_region_image.dtype)
    indices = np.argwhere(selected_region_image == 1)
    for index in indices:
        row, col = index
        bordered_arr[row:row + border_pixels, col:col + border_pixels] = 1
        bordered_arr[row - border_pixels:row, col - border_pixels:col] = 1

   
    norm_old_method= (dicom_slice1 - np.min(dicom_slice1)) / (np.max(dicom_slice1) - np.min(dicom_slice1))
    artery_nn=norm_old_method*bordered_arr 
    artery_nn[0,0] = 1
    #artery_new = (artery - artery.min()) / (artery.max() - artery.min()) * 255
    #artery_8bit = artery_new.astype('uint8')
   

    min_val=dicom_slice.max()-value
    max_val=dicom_slice.max()
    combined_normalized_image2=(np.clip(dicom_slice, min_val, max_val) - min_val) / (max_val - min_val)
    combined_normalized_image2_8bit = (combined_normalized_image2 * 255).astype(np.uint8)


    custom_config = r'--oem 3 --psm 6'
    text_boxes = pytesseract.image_to_data(combined_normalized_image2_8bit, output_type=pytesseract.Output.DICT, config=custom_config)
    bounding_boxes = []

    highest_confidence = -1
    best_bounding_box = None

    # Iterate over detected text boxes
    for i in range(len(text_boxes['text'])):
        conf = int(text_boxes['conf'][i])
        word = clean_text(text_boxes['text'][i])
        print(word)
        #print(f"Word before cleaning: '{text_boxes['text'][i]}'")  # Print the cleaned word
        #print(f"Word in uppercase: '{word.upper()}'")  # Print the word in uppercase
        # Skip this iteration if there's no actual text
        if word == '':
            #print(word)
            continue

        (x, y, w, h) = (text_boxes['left'][i], text_boxes['top'][i], text_boxes['width'][i], text_boxes['height'][i])
    
        if word in ['LAD', 'LAD_', 'RCA', 'LCX', 'CX', 'LAD_2', 'LAD_13','RCA_15','RCA_med', 'pl','med','rpd','RCAmed']:
            print(f"Detected text: {text_boxes['text'][i]} at position {(x, y, w, h)}")
            bounding_boxes.append((x, y, w, h))
        else:
            #print(f"Detected other text: {word} (Confidence: {conf}) at position {(x, y, w, h)}")
            # Store the bounding box and keep track of the one with the highest confidence
            if conf > highest_confidence:
                highest_confidence = conf
                best_bounding_box = (x, y, w, h)

    if not bounding_boxes and best_bounding_box:
        bounding_boxes.append(best_bounding_box)

    if not bounding_boxes:
        print("No matching text found. Displaying the original image.")
        print(file_path)
        filled_image = artery_nn  # Use the original image without any modifications
    else:
        #artery_8bit = (artery * 255).astype(np.uint8) if artery.max() <= 1 else artery.astype(np.uint8)
        filled_image = fill_bounding_box(artery_nn, bounding_boxes, window_size=10)

    return filled_image

artery=fetch_artery_with_bounday_all_slices_3ch_new(class_1_patient_ids[:], class_1_dicom_ids[:], class_1_dicom_fold_num[:],class_1_destination_dir,artery_to_process)
artery=fetch_artery_with_bounday_all_slices_3ch_new(class_2_patient_ids[:], class_2_dicom_ids[:], class_2_dicom_fold_num[:],class_2_destination_dir,artery_to_process)

