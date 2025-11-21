import os
import shutil
# import pandas as pd
def get_mp4_files(root_dir, max_depth=2):
    mp4_files = []
    
    def scan_directory(path, current_depth=0):
        if current_depth >= max_depth:
            return
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith('.mp4'):
                            mp4_files.append(entry.path)  # Store full path
                        elif entry.is_dir(follow_symlinks=False) and current_depth + 1 < max_depth:
                            scan_directory(entry.path, current_depth + 1)
                    except (PermissionError, OSError):
                        continue  # Skip entries with access issues
        except (PermissionError, OSError) as e:
            print(f"Error accessing {path}: {e}")

    scan_directory(root_dir)
    return sorted(mp4_files)


if __name__ == "__main__":
    import tqdm 
    import glob
    import sys

    # Add the absolute path to the ./src folder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'src/PyThon/FFMpeg')))
    from Jpg2Video import process_experiment


    adress = "/media/ubun25/DONT/MPI/S4S-ROF/drop/"
    experiments = get_mp4_files(adress, max_depth=5)
    experiments = [experiment.replace("drop","frame_Extracted") for experiment in experiments]
    experiments = [experiment.replace(".mp4","") for experiment in experiments]
    experiments = [experiment.replace("frames","frame_Extracted") for experiment in experiments]

    for exp in tqdm.tqdm(experiments):
        destination = exp.replace("frame_Extracted","4S-SROF")
        if not os.path.isdir(destination):
            os.makedirs(destination, exist_ok=True)

        video       = os.path.join(exp,'SR_edge','result.mp4')
        if os.path.isfile(video):
            shutil.move(video,destination)
            if (len(glob.glob(os.path.join(exp,'SR_edge','*.jpg')))>0):
                process_experiment(exp)
                os.remove(os.path.join(destination,'result.mp4'))
                shutil.move(video,destination)
            os.removedirs(os.path.join(exp,'SR_edge'))

        CSV         = os.path.join(exp,'SR_result','result.csv')
        if os.path.isfile(CSV):
            shutil.move(CSV,destination)
            os.removedirs(os.path.join(exp,'SR_result'))

        # break

