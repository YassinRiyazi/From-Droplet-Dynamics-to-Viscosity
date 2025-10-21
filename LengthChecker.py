"""
[Delegated to AVP library]

Author:         Yassin Riyazi
Date:           25.08.2025
Description:    This script checks the lengths of experimental directories and removes those that are significantly shorter than average.
License:        GNU General Public License v3.0

"""
import  os
import  glob
import  send2trash # type: ignore
import  subprocess # type: ignore
import  numpy       as      np
from    typing      import  Dict, Union


def delete_empty_dirs(root_dir:Union[str, os.PathLike[str]]) -> None:
    # Walk bottom-up so we remove empty sub-folders first
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:  # If folder has no subdirs and no files
            print(f"Deleting empty folder: {dirpath}")
            os.rmdir(dirpath)

if __name__ == "__main__":
    HauptAddress = '/media/d2u25/Dont/frames_Process_30'
    # delete_empty_dirs(HauptAddress)

    for tilt in sorted(glob.glob(f"{HauptAddress}/*")):
        
        for experiment in sorted(glob.glob(os.path.join(tilt, '*'))):
            length: Dict[str, int] = {}
            
            for rep in sorted(glob.glob(os.path.join(experiment, '*'))):
                length[rep] = len(glob.glob(os.path.join(rep, '*.png')))

            Average = np.mean(np.array(list(length.values())))
            std = np.std(np.array(list(length.values())))

            addresses: list[str] = []
            for rep, value in length.items():
                if np.abs(value - Average) > 0.25 * Average:
                    print(f"{rep} is {np.abs(value - Average)*100/Average:.2f}%, {std:.2f}")
                    addresses.append(os.path.split(rep)[0])
                    # send2trash.send2trash(rep)
                else:
                    pass
            
            addresses = list(dict.fromkeys(addresses))
            for address in addresses:
                subprocess.run(["xdg-open", address])
