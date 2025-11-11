import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import glob, os

from numpy.typing import NDArray
import re

def build_dataset(addresses: list[str]) -> NDArray[np.float64]:
    data = np.zeros((len(addresses), 3), dtype=np.float64)
    for idx, adress in enumerate(addresses):
        spil = adress.split(os.sep)
        data[idx,0] = len(glob.glob(f"{adress}/*.png"))
        data[idx,1] = (360 - int(spil[-4]))*np.pi/180

        data[idx,2] = float(spil[-2].split('_')[-1])/127
    return data
if __name__ == "__main__":
    data_dir = '/media/d25u2/Dont/Teflon-AVP'

    tilt_folders = sorted(glob.glob(f'{data_dir}/*',))

    # check only folders with tilt
    tilt_folders = [folder for folder in tilt_folders if re.match(r'.*/-?\d+$', folder)]
    addresses = [glob.glob(f"{tilt_folder}/*/*/databases") for tilt_folder in tilt_folders]
    addresses = [item for sublist in addresses for item in sublist]

    data_train = build_dataset(addresses)

    x_train = data_train[:, 0:2]
    y_train = data_train[:, 2]

    linreg = LinearRegression()
    linreg.fit(x_train, y_train)

    # Validation on training set
    y_pred_val_linear = linreg.predict(x_train)
    mae_linear = np.mean(np.abs(y_train - y_pred_val_linear))
    print("Linear Fit MAE on validation:", mae_linear,mae_linear*mae_linear)


# x_val = data_val[:, 1:2]
# y_val = data_val[:, 2]

# # --- Linear Fit ---
# linreg = LinearRegression()
# linreg.fit(x_train, y_train)

# y_pred_val_linear = linreg.predict(x_val)
# mae_linear = np.mean(np.abs(y_val - y_pred_val_linear))
# print("Linear Fit MAE on validation:", mae_linear,mae_linear*mae_linear)


"""
Only tilt: Linear Fit MAE on validation: 0.19030730522739361 0.036216870422912356
Only time: Linear Fit MAE on validation: 0.06893702595204221 0.004752313547112541
Tilt & time: Linear Fit MAE on validation: 0.05519721910693014 0.0030467329971384533
"""