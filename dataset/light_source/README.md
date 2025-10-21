# Light Source Reflection Removal
![`Sample`](../../Samples/frame_000514.png)  
Original Image


Small utility module to detect and remove light-source reflections from grayscale images.

Main file: [dataset/lightSource/LightSourceReflectionRemoving.py](LightSourceReflectionRemoving.py)

Core functions
- [`dataset.lightSource.LightSourceReflectionRemoving.DropBoundaryExtractor`](LightSourceReflectionRemoving.py)  
  Extracts the main drop/reflection boundary from a gray-scale image and writes a masked image to disk.
  ![`Sample`](./doc/DropBoundryExtractor.png)  

- [`dataset.lightSource.LightSourceReflectionRemoving.LightSourceReflectionRemover`](LightSourceReflectionRemoving.py)  
  Produces a binary mask that removes the light-source reflection area; configurable via `threshold_activation`.
  ![`Sample`](./doc/LightSourceReflectionRemover.png)  
  

Quick usage
- Run the module directly to produce two samples in `__main__`:  
  ```sh
  python dataset/lightSource/LightSourceReflectionRemoving.py
  ```

API notes
- Input images are read with OpenCV in gray scale. Functions raise on unreadable files.
- `LightSourceReflectionRemover` uses contour detection on a blurred, inverted image to find and mask the brightest reflection area. Adjust `threshold_activation` to tune mask inversion behavior.
- `process_image` is designed for use with multiprocessing pools; `process_images_in_directory` currently prepares directories and runs removals sequentially but the `__main__` includes an example of parallel processing using `multiprocessing.Pool`.

Dependencies
- Python 3.11+
- OpenCV (cv2), numpy

Related files
- LICENSE: [LICENSE](LICENSE)
- Source implementation: [dataset/lightSource/LightSourceReflectionRemoving.py](LightSourceReflectionRemoving.py)
- Dataset: [dataset/DaughterFolderDataset.py](../DaughterFolderDataset.py)

Notes and tips
- Test on a few images before processing the entire dataset; inspect masks visually.
- For large datasets, enable the multiprocessing block in `__main__` and tune `chunksize` and worker count.