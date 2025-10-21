"""
Author:         Yassin Riyazi
Date:           20.08.2025
Description:    Remove light source reflection from an image.
License:        General Public License v3.0
"""
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import List, Tuple, Union
from tqdm import tqdm


def DropBoundaryExtractor(imageAddress:str,
                         outputAddress:str,) -> None:
    """
    Returning the boundary of drop.
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/lightSource/doc/DropBoundaryExtractor.png" alt="Italian Trulli">
    
    Args:
        imageAddress (str): The file path to the input image.
        outputAddress (str): The file path to save the output image.
        
    Returns:
        None: none
    """

    # Load the image
    source_img = cv2.imread(imageAddress, cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        raise ValueError("Image not found or unable to load.")

    # Invert the image
    inverted_img = cv2.bitwise_not(source_img)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(inverted_img, (5, 5), 0)

    # Threshold to create a binary mask
    _, binary_mask = cv2.threshold(blurred_img, 10, 250, cv2.THRESH_BINARY_INV)

    # Find contours of the light source reflection
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(source_img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Invert the mask to get the area without reflection
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    result_img = cv2.bitwise_and(source_img, source_img, mask=mask_inv)

    # Save or display the result
    cv2.imwrite(outputAddress, result_img)

def LightSourceReflectionRemover(imageAddress:str,
                                 outputAddress:str,
                                 threshold_activation:int = 100) -> None:
    """
    Remove light source reflection from an image.
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/lightSource/doc/LightSourceReflectionRemover.png" alt="Italian Trulli">
    
    Args:
        imageAddress (str): The file path to the input image.
        outputAddress (str): The file path to save the output image.
        threshold_activation (int, optional): The threshold value for reflection removal. Defaults to 100.

    Returns:
        None: none

    TODO:
        - Add assertion for input data type and value.
    """
    # Load the image
    source_img = cv2.imread(imageAddress, cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        raise ValueError("Image not found or unable to load.")

    # Invert the image
    inverted_img = cv2.bitwise_not(source_img)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(inverted_img, (5, 5), 0)

    # Threshold to create a binary mask
    _, binary_mask = cv2.threshold(blurred_img, 10, 250, cv2.THRESH_BINARY_INV)

    # Find contours of the light source reflection
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(source_img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    inside = (mask <= threshold_activation).astype(np.uint8) * 255
    inside = cv2.bitwise_not(inside)

    # Save or display the result
    cv2.imwrite(outputAddress, inside)

def process_image(args:tuple) -> None:
    """Helper for parallel processing."""
    image, output_dir = args

    output_dir = os.path.join(output_dir, os.path.basename(image))
    if not os.path.isfile(output_dir):
        LightSourceReflectionRemover(image, output_dir)
    # return image  # for logging if needed


if __name__ == "__main__":
    ImageAddress = 'src/PyThon/ContactAngle/DropSamples/frame_000514.png'
    DropBoundaryExtractor(ImageAddress,
                         'src/PyThon/Viscosity/lightSource/doc/DropBoundryExtractor.png')
    
    LightSourceReflectionRemover(ImageAddress,
                                 'src/PyThon/Viscosity/lightSource/doc/LightSourceReflectionRemover.png')