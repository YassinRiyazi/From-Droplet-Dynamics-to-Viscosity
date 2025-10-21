"""
[Delegated to AVP library]

Author:     Yassin Riyazi
Date:       04.08.2025
Description:
    This script processes directories containing images of drops, extracting and analyzing them using the DropDetection_Sum module.
    It utilizes multiprocessing to handle multiple directories concurrently and employs tqdm for progress tracking.
    It is designed to work with a specific directory structure and image processing requirements.
    It is part of the P2NeuralNetwork project under the Viscosity project umbrella.
License:    GNU General Public License v3.0

"""
import  glob
import  os
# import  cv2
# import  numpy                   as      np
# import  matplotlib.pyplot       as      plt
from    multiprocessing         import  Pool
from    tqdm                    import  tqdm
from   typing                  import  Dict,Union
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/ContactAngle/DropDetection')))
from DropDetection_Sum import Main # type: ignore

import Nphase4_AutoEncoder.dataset as DSS

def worker(args:Dict[str, Union[str, bool]]):
    """Worker function to call Main with the given arguments."""
    try:
        Main(
            experiment=args['experiment'],
            SaveAddress=args['SaveAddress'],
            SaveAddressCSV=args['SaveAddressCSV'],
            extension=args['extension'],
            _morphologyEx=args['_morphologyEx']
        )
    except Exception as e:
        print(f"Error processing {args['experiment']}: {e}")

if __name__ == "__main__":
    Name = 'Velocity_P540'#'Velocity_wide'
    
    # # Set the number of processes (adjust as needed)
    fps = 30
    num_processes = 14  # Example: Use 4 processes
    # Collect all tasks
    tasks = []

    HauptAddress = '/media/d2u25/Dont/frames_Process_30'
    for tilt in glob.glob(f"{HauptAddress}/*"):
        for experiment in glob.glob(os.path.join(tilt, '*')):
            for _idx, rep in enumerate(glob.glob(os.path.join(experiment, '*'))):
                # if _idx < 5:
                _SaveAddresses = rep.replace(f'frames_Process_{fps}', f'frames_Process_{fps}_{Name}')
                tasks.append({
                    'experiment': rep,
                    'SaveAddress': _SaveAddresses,
                    'SaveAddressCSV': _SaveAddresses,
                    'extension': '.png',
                    '_morphologyEx': True
                })

    # data_dir = '/media/d2u25/Dont/frames_Process_30_Velocity'
    # dicAddressesTrain, dicAddressesValidation, dicAddressesTest = DSS.dicLoader()
    # del dicAddressesTest

    # for key in dicAddressesTrain:
    #     ValidPaths = [os.path.join(data_dir, os.path.relpath(path, '/media/d2u25/Dont/frames_Process_30')) for path in dicAddressesTrain[key]]
    #     dicAddressesTrain[key] = ValidPaths

    # for key in dicAddressesValidation:
    #     ValidPaths = [os.path.join(data_dir, os.path.relpath(path, '/media/d2u25/Dont/frames_Process_30')) for path in dicAddressesValidation[key]]
    #     dicAddressesValidation[key] = ValidPaths

    # for key in dicAddressesTrain:
    #     for idx in range(len(dicAddressesTrain[key])):
    #         tasks.append({
    #             'experiment': dicAddressesTrain[key][idx].replace(f'frames_Process_{fps}_Velocity',f'frames_Process_{fps}'),
    #             'SaveAddress': dicAddressesTrain[key][idx],
    #             'SaveAddressCSV': dicAddressesTrain[key][idx],
    #             'extension': '.png',
    #             '_morphologyEx': True
    #         })

    # for key in dicAddressesValidation:
    #     for idx in range(len(dicAddressesValidation[key])):
    #         print(f"Preparing task for: {dicAddressesValidation[key][idx]}")
    #         tasks.append({
    #             'experiment': dicAddressesValidation[key][idx].replace(f'frames_Process_{fps}_Velocity',f'frames_Process_{fps}'),
    #             'SaveAddress': dicAddressesValidation[key][idx],
    #             'SaveAddressCSV': dicAddressesValidation[key][idx],
    #             'extension': '.png',
    #             '_morphologyEx': True
    #         })

    # Run tasks in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Processing videos"))

    # """
    #     Check the YOLO result with OpenCV vcountors
    #     Normalize the white lines in bottom of the images
    #     save x1 in the textfile with same name as the image
    # """
