import  os
import  glob
from    tqdm            import  tqdm
from    multiprocessing import  Pool
from   typing          import  Dict, List, Tuple
import  sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/Viscosity/PositionalEncoding')))
from PossitionalImageGenerator import make_PE_image_Folder, make_PE_image_FolderFullScale #type: ignore

def worker_Wide(args: Dict[str, object]) -> None:
    try:
        make_PE_image_FolderFullScale(address =              args['address'],
                             verbose =              args['verbose'],
                             extension =            args['extension'],
                             remove_Previous_Dir =  args['remove_Previous_Dir'],
                             velocity_encoding   =  args['velocity_encoding'],
                             positional_encoding =  args['positional_encoding'])
    except Exception as e:
        print(f"Failed to process {args['address']}: {e}")
    
    return None

def worker(args: Dict[str, object]) -> None:
    try:
        make_PE_image_Folder(address =              args['address'],
                             verbose =              args['verbose'],
                             extension =            args['extension'],
                             remove_Previous_Dir =  args['remove_Previous_Dir'],
                             velocity_encoding   =  args['velocity_encoding'],
                             positional_encoding =  args['positional_encoding'])
    except Exception as e:
        print(f"Failed to process {args['address']}: {e}")

    return None

def list_Gen(FolderName: str) -> Tuple[List[Dict[str, object]], str]:
    HaupftAddress = f'/media/roboprocessing/Data/{FolderName}' 
    if 'velocity' in str.lower(FolderName):
        velocity_encoding = True
        positional_encoding = False
    elif 'position' in str.lower(FolderName):
        velocity_encoding = False
        positional_encoding = True
    else:
        raise ValueError("FolderName must contain either 'velocity' or 'positional' to determine encoding type.")

    tasks:List[Dict[str, object]] = []
    for tilt in sorted(glob.glob(os.path.join(HaupftAddress, '*'))):
        for exp in sorted(glob.glob(os.path.join(tilt, '*'))):
            for rep in sorted(glob.glob(os.path.join(exp, '*'))):
                tasks.append({
                        'address': rep,
                        'verbose': False,
                        'extension': '.png',
                        'remove_Previous_Dir': False,
                        'velocity_encoding':   velocity_encoding,#     False,
                        'positional_encoding': positional_encoding,#     True
                    })
    return tasks, HaupftAddress           


if __name__ == "__main__":
    # FolderName = 'frames_Process_30_Velocity'#'frames_Process_30_Velocity_wide'
   
    for FolderName in ['frames_Process_30_Velocity_P540',
                       ]:
        tasks, HaupftAddress = list_Gen(FolderName)
        num_processes = 14  # Example: Use 14 processes
        if 'wide' in str.lower(HaupftAddress):        
            with Pool(processes=num_processes) as pool:
                list(tqdm(pool.imap(worker_Wide, tasks), total=len(tasks), desc="Processing Wide videos"))
        else:
            with Pool(processes=num_processes) as pool:
                list(tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Processing videos"))