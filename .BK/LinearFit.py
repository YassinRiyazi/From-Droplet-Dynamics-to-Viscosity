import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import glob, os
import dataset as DSS

# --- Prepare training data ---
stride = 4
sequence_length = 1 
data_dir = '/media/roboprocessing/Data/frames_Process_30_Velocity_P540'
dicAddressesTrain, dicAddressesValidation, dicAddressesTest = DSS.dicLoader(root=data_dir)
del dicAddressesTest

def build_dataset(addresses):
    data = np.zeros((len(addresses), 3), dtype=np.float64)
    for idx, adress in enumerate(addresses):
        spil = adress.split(os.sep)
        data[idx,0] = len(glob.glob(f"{adress}/*.png"))
        data[idx,1] = 360 - int(spil[-3])
        data[idx,2] = float(spil[-1].split('_')[-1])/127
    return data

# Train dataset
addresses_train = []
for key in dicAddressesTrain.keys():
    addresses_train.extend(dicAddressesTrain[key])
data_train = build_dataset(addresses_train)

x_train = data_train[:, 1:2]
y_train = data_train[:, 2]

# Validation dataset
addresses_val = []
for key in dicAddressesValidation.keys():
    addresses_val.extend(dicAddressesValidation[key])
data_val = build_dataset(addresses_val)

x_val = data_val[:, 1:2]
y_val = data_val[:, 2]

# --- Linear Fit ---
linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_pred_val_linear = linreg.predict(x_val)
mae_linear = np.mean(np.abs(y_val - y_pred_val_linear))
print("Linear Fit MAE on validation:", mae_linear,mae_linear*mae_linear)


"""
Only tilt: Linear Fit MAE on validation: 0.19030730522739361 0.036216870422912356
Only time: Linear Fit MAE on validation: 0.06893702595204221 0.004752313547112541
Tilt & time: Linear Fit MAE on validation: 0.05519721910693014 0.0030467329971384533
"""