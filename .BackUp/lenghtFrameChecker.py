"""
[Delegated to AVP library]
Author:         Yassin Riyazi
Date:           21.10.2025
Description:    Checking length of video frames by calculating number of images inside a folder
License:        GNU General Public License v3.0
"""
import os
import glob
import numpy as np

if __name__ == "__main__":
    for tilt in sorted(glob.glob("/media/d2u25/Dont/frames_Process_30/*")):
        for experiment in sorted(glob.glob(os.path.join(tilt,'*'))):
            length: dict[str, int] = {}
            for rep in sorted(glob.glob(os.path.join(experiment,'*'))):
                length[rep] = os.listdir(rep).__len__()

            avg = np.array(list(length.values())).mean()
            for rep, count in length.items():
                if np.abs(count - avg)/avg > 0.2:
                    print(f"Warning: {rep} has {count} frames, which deviates significantly from the average of {avg:.2f} frames.")
                    break
            