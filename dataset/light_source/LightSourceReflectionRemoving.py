"""
Author:         Yassin Riyazi
Date:           20.08.2025
Description:    Remove light source reflection from an image.
License:        General Public License v3.0

Learned:
    1.  If you simply differentiate two images, the result may wrap around due to unsigned integer representation. Therefore a noisy result.
        To mitigate this, use cv2.subtract() or convert images to a signed integer type before subtraction. 
            diff            = vv.astype(np.int16) - img.astype(np.int16)
            diff_clipped    = np.clip(diff, 0, 255).astype(np.uint8)
        This prevents wrap-around and ensures that negative differences are represented correctly.
"""
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional

def DropBoundaryExtractor(imageAddress:NDArray[np.int8] | str,
                         outputAddress:str|None = None,
                         ) -> NDArray[np.int8]:
    """
    Returning the boundary of drop.
    
    Args:
        imageAddress (str): The file path to the input image.
        outputAddress (str): The file path to save the output image.
        
    Returns:
        None: none
    """
    if isinstance(imageAddress, str):
        source_img = cv2.imread(imageAddress, cv2.IMREAD_GRAYSCALE)
        if source_img is None:
            raise ValueError("Image not found or unable to load.")
    else:
        source_img = imageAddress

    # Invert the image
    assert len(source_img.shape) == 2, "Input image must be a grayscale image."

    # if source_img[-1,:].mean() < 250:
    # source_img = cv2.bitwise_not(source_img)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(source_img, (5, 5), 0)

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
    if outputAddress is not None:
        # Save or display the result
        cv2.imwrite(outputAddress, result_img)
    return result_img

def select_inner_region(
    image: NDArray[np.uint8] | str,
    method: str = "distance",
    inner_fraction: float = 0.5,
    morph_kernel_size: int = 7,
    erosion_iters: Optional[int] = None,
    blur_ksize: int = 5,
    debug: bool = False
) -> Dict[str, NDArray[np.uint8]]:
    """
    Detect the largest object (outer contour) in a grayscale image and return a clean inner region mask.

    Args:
        image: Grayscale image array (H,W) or a path to an image file.
        method: "distance" (default) uses distance transform to pick inner core;
                "erode" uses morphological erosion to get inner region.
        inner_fraction: For method="distance", keep pixels with distance > inner_fraction * max_distance.
                        Value in (0,1). Higher -> smaller inner core.
        morph_kernel_size: Kernel size for morphological opening/closing (odd integer).
        erosion_iters: If using method="erode", number of erosions. If None, a heuristic is used.
        blur_ksize: Gaussian blur kernel size (odd).
        debug: If True, returns additional debug images in the dict and prints sizes.

    Returns:
        A dict with keys:
          - "mask": filled mask of largest contour (uint8: 0/255)
          - "inner_mask": inner mask (uint8: 0/255)
          - "inner_region": original grayscale with inner_mask applied
          - "outer_removed": original grayscale with the outer region removed (i.e., background + inner kept or inverted, see below)
          - if debug: "binary", "cleaned" masks included too.

    Notes:
        - The function will invert the image if the border is bright (assumes object is darker/lighter accordingly).
        - If no contour is found, returns zero masks.
    """
    # --- load and validate
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from path: {image}")
    else:
        img = image.copy()

    if img.ndim != 2:
        raise AssertionError("Input must be a grayscale image (2D array).")

    h, w = img.shape

    # --- decide if we must invert (if border is bright, invert so object is white)
    # compute mean of a small border (10 px or less)
    border = 10
    border = min(border, h//4, w//4)
    top = img[:border, :]
    bottom = img[-border:, :]
    left = img[:, :border]
    right = img[:, -border:]
    border_mean = float(np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()]).mean())

    img_proc = img.copy()
    if border_mean > 200:  # likely white background -> invert to make object white
        img_proc = cv2.bitwise_not(img_proc)

    # --- blur and threshold (use Otsu to be adaptive)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(img_proc, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- morphological cleanup
    k = max(3, morph_kernel_size // 2 * 2 + 1)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- find largest contour (assumed to be the drop)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        empty = np.zeros_like(img, dtype=np.uint8)
        return {
            "mask": empty,
            "inner_mask": empty,
            "inner_region": empty,
            "outer_removed": empty,
            **({"binary": binary, "cleaned": cleaned} if debug else {})
        }

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # --- get inner mask
    if method == "distance":
        # Distance transform on the filled mask (convert to 0/1 for transform)
        mask_bin = (mask // 255).astype(np.uint8)
        # distance transform requires foreground as non-zero
        dist = cv2.distanceTransform(mask_bin, distanceType=cv2.DIST_L2, maskSize=5)
        max_d = dist.max() if dist.size else 0.0
        if max_d <= 0:
            inner_mask = np.zeros_like(mask)
        else:
            # threshold on fraction of max distance
            th = inner_fraction * max_d
            inner_mask = (dist > th).astype(np.uint8) * 255

    elif method == "erode":
        # determine iterations if not provided
        if erosion_iters is None:
            # heuristic: erode until area reduces to about inner_fraction of original
            total_area = cv2.countNonZero(mask)
            target_area = int(total_area * inner_fraction)
            if target_area <= 0:
                erosion_iters = 1
            else:
                # iteratively erode until under target or max 50
                inner_mask = mask.copy()
                it = 0
                while cv2.countNonZero(inner_mask) > target_area and it < 50:
                    inner_mask = cv2.erode(inner_mask, kernel, iterations=1)
                    it += 1
                erosion_iters = it
        inner_mask = cv2.erode(mask, kernel, iterations=erosion_iters)
    else:
        raise ValueError("method must be 'distance' or 'erode'")

    # --- final masked images
    inner_region = cv2.bitwise_and(img, img, mask=inner_mask)
    outer_removed = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    out = {
        "mask": mask,
        "inner_mask": inner_mask,
        "inner_region": inner_region,
        "outer_removed": outer_removed
    }
    if debug:
        out["binary"] = binary
        out["cleaned"] = cleaned
        print(f"image: {w}x{h}, border_mean={border_mean:.1f}, largest_area={cv2.contourArea(largest):.0f}, "
              f"inner_npix={cv2.countNonZero(inner_mask)}")
    return out

def LightSourceReflectionRemover(image:NDArray[np.int8] | str,
                                 outputAddress:str | None = None,
                                 threshold_activation:int = 100) -> NDArray[np.int8]:
    """
    Remove light source reflection from an image.
    
    Args:
        imageAddress (str): The file path to the input image.
        outputAddress (str): The file path to save the output image.
        threshold_activation (int, optional): The threshold value for reflection removal. Defaults to 100.

    Returns:
        None: none

    TODO:
        - Add assertion for input data type and value.
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found or unable to load.")

    # Invert the image
    assert len(image.shape) == 2, "Input image must be a grayscale image."
    assert 0 <= threshold_activation <= 255, "Threshold activation must be in the range [0, 255]."

    flip = False
    if image[-1,:].mean() < 250:
        image = cv2.bitwise_not(image)
        flip = True

    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold to create a binary mask
    _, binary_mask = cv2.threshold(image, 10, 250, cv2.THRESH_BINARY_INV)

    # Find contours of the light source reflection
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    inside = (mask <= threshold_activation).astype(np.uint8) * 255
    # inside = cv2.bitwise_not(inside)
    
    if outputAddress is not None:
        cv2.imwrite(outputAddress, inside)

    if flip:
        inside = cv2.bitwise_not(inside)
    return inside

if __name__ == "__main__":
    img = cv2.imread("/media/Dont/Teflon-AVP/280/S3-SDS10_D/T110_06_0.900951687825/databases/frame_000348.png", cv2.IMREAD_GRAYSCALE)
    vv = LightSourceReflectionRemover(img)

    # using cv2 bitwise and to remove the light source reflection
    cv2.imshow("Original", img)
    cv2.imshow("LightSourceReflectionRemover", vv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # safe signed difference (no wrap)
    diff = vv.astype(np.int16) - img.astype(np.int16)
    diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)
    cv2.imshow("Result1", diff_clipped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    res = select_inner_region(diff_clipped, method="distance", inner_fraction=0.009, morph_kernel_size=3, debug=True)

    cv2.imshow("filled_mask", res["mask"])
    cv2.imshow("inner_mask", res["inner_mask"])
    # cv2.imshow("inner_region", res["inner_region"])
    cv2.imshow("outer_removed", res["outer_removed"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()