# Light Source Reflection Removal

Small utility module to detect and remove light-source reflections from grayscale images.

Main file: [dataset/lightSource/LightSourceReflectionRemoving.py](dataset/lightSource/LightSourceReflectionRemoving.py)

Core functions
- [`dataset.lightSource.LightSourceReflectionRemoving.DropBoundaryExtractor`](dataset/lightSource/LightSourceReflectionRemoving.py)  
  Extracts the main drop/reflection boundary from a gray-scale image and writes a masked image to disk.

- [`dataset.lightSource.LightSourceReflectionRemoving.LightSourceReflectionRemover`](dataset/lightSource/LightSourceReflectionRemoving.py)  
  Produces a binary mask that removes the light-source reflection area; configurable via `threshold_activation`.
  
- [`dataset.lightSource.LightSourceReflectionRemoving.process_image`](dataset/lightSource/LightSourceReflectionRemoving.py)  
  Helper wrapper for parallel worker tasks (image, output_dir) → processes one image.

- [`dataset.lightSource.LightSourceReflectionRemoving.process_images_in_directory`](dataset/lightSource/LightSourceReflectionRemoving.py)  
  High-level helper that builds input→output directory map for dataset directories and runs the remover sequentially.

Quick usage
- Run the module directly to process dataset folders (parallel mode is implemented but commented/adjustable in the `__main__`):  
  ```sh
  python dataset/lightSource/LightSourceReflectionRemoving.py
  ```
  The script looks for folders like `/media/d2u25/Dont/frames_Process_<FPS>/...` and creates corresponding `frames_Process_<FPS>_LightSource` output folders.

API notes
- Input images are read with OpenCV in gray scale. Functions raise on unreadable files.
- `LightSourceReflectionRemover` uses contour detection on a blurred, inverted image to find and mask the brightest reflection area. Adjust `threshold_activation` to tune mask inversion behavior.
- `process_image` is designed for use with multiprocessing pools; `process_images_in_directory` currently prepares directories and runs removals sequentially but the `__main__` includes an example of parallel processing using `multiprocessing.Pool`.

Dependencies
- Python 3.8+
- OpenCV (cv2), numpy, tqdm, matplotlib

Related files
- LICENSE: [LICENSE](LICENSE)
- Source implementation: [dataset/lightSource/LightSourceReflectionRemoving.py](dataset/lightSource/LightSourceReflectionRemoving.py)
- Dataset usage

Notes and tips
- Test on a few images before processing the entire dataset; inspect masks visually.
- For large datasets, enable the multiprocessing block in `__main__` and tune `chunksize` and worker count.