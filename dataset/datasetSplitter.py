"""
    Author: Yassin Riyazi
    Date: 25-08-2025
    Description: This script splits a dataset into training, validation, and test sets.

    Learned:
        1. TypeAlias: to define a new type alias
            # Define the type alias
            StringListDict: TypeAlias = Dict[str, List[str]]

"""

import os
import glob
import pickle
from typing import Dict, Union, List, TypeAlias
# Define the type alias
StringListDict: TypeAlias = Dict[str, List[str]]


def delete_empty_dirs(root_dir:Union[str, os.PathLike[str]]) -> None:
    # Walk bottom-up so we remove empty sub-folders first
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:  # If folder has no subdirs and no files
            print(f"Deleting empty folder: {dirpath}")
            os.rmdir(dirpath)


def dicMaker(HauptAddress:Union[str, os.PathLike[str]] = '/media/roboprocessing/Data/frames_Process_30_Position',
             BooldelEmpty:bool = False,
             BoolReturn:bool = False) -> None|tuple[StringListDict, StringListDict, StringListDict]:
    if BooldelEmpty:
       delete_empty_dirs(HauptAddress)

    fluid: set[str] = set()
    for tilt in sorted(glob.glob(f"{HauptAddress}/*")):
        for experiment in sorted(glob.glob(os.path.join(tilt, '*'))):
            fluid.add(os.path.basename(experiment))

    dicAddressesTrain:StringListDict = {f: [] for f in fluid}
    dicAddressesValidation:StringListDict = {f: [] for f in fluid}
    dicAddressesTest:StringListDict = {f: [] for f in fluid}


    for tilt in sorted(glob.glob(f"{HauptAddress}/*")):
        for experiment in sorted(glob.glob(os.path.join(tilt, '*'))):

            repetitionNumber = sorted(glob.glob(os.path.join(experiment, '*')))
            for idx, rep in enumerate(repetitionNumber):
                rep = os.path.relpath(rep,HauptAddress)
                if idx <= 5 and len(repetitionNumber) >= 6:
                    dicAddressesTrain[os.path.basename(experiment)].append(rep)
                elif idx > 5 and idx <= 3 + 5:
                    dicAddressesValidation[os.path.basename(experiment)].append(rep)
                else:
                    dicAddressesTest[os.path.basename(experiment)].append(rep)

    rootAddress = os.path.dirname(__file__)
    pickle.dump(dicAddressesTrain, open(os.path.join(rootAddress, "dicAddressesTrain.pkl"), "wb"))
    pickle.dump(dicAddressesValidation, open(os.path.join(rootAddress, "dicAddressesValidation.pkl"), "wb"))
    pickle.dump(dicAddressesTest, open(os.path.join(rootAddress, "dicAddressesTest.pkl"), "wb"))

    if BoolReturn:
        return dicAddressesTrain, dicAddressesValidation, dicAddressesTest

def pathCompleterList(paths:List[Union[str, os.PathLike[str]]],
                      root:Union[str, os.PathLike[str]]) -> List[str]:
    return [os.path.join(root, path) for path in paths]

def dicLoader(rootAddress:Union[None,str, os.PathLike[str]]=None,
              root:Union[None,str, os.PathLike[str]]=None) -> tuple[StringListDict, StringListDict, StringListDict]:
    if rootAddress is None:
        rootAddress = os.path.dirname(__file__)
    dicAddressesTrain       = pickle.load(open(os.path.join(rootAddress, "dicAddressesTrain.pkl"),       "rb"))
    dicAddressesValidation  = pickle.load(open(os.path.join(rootAddress, "dicAddressesValidation.pkl"),  "rb"))
    dicAddressesTest        = pickle.load(open(os.path.join(rootAddress, "dicAddressesTest.pkl"),        "rb"))

    if root is None:
        return dicAddressesTrain, dicAddressesValidation, dicAddressesTest
    
    for dic in [dicAddressesTrain, dicAddressesValidation, dicAddressesTest]:
        for key in dic.keys():
            dic[key] = pathCompleterList(dic[key], root)

    return dicAddressesTrain, dicAddressesValidation, dicAddressesTest

if __name__ == "__main__":
    dicAddressesTrain, dicAddressesValidation, dicAddressesTest = dicLoader()
