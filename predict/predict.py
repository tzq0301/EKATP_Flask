import os

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame


def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1], 1)
    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    # return "dimenional error"
    raise ValueError("dimensional error")


def predict_(model, datapath: str, result_folder, step: int = 6):
    if step is None or step == 0:
        step = 6

    # ******************************************************************************
    # load data
    # ******************************************************************************
    X = pd.read_csv(datapath)
    X = X.T
    X = np.array(X)

    xmin = np.min(X)
    xptp = np.ptp(X)
    # scale
    X = 2 * (X - xmin) / xptp - 1

    X = add_channels(X)
    X = torch.from_numpy(X).float().contiguous()

    Xinput = X[:-1]

    # ******************************************************************************
    # Prediction
    # ******************************************************************************
    init = Xinput[0].float()
    preds = []

    z = model.encoder(init)

    for j in range(step):
        if isinstance(z, tuple):
            z = model.dynamics(*z)
            x_pred = model.decoder(z[0])
        else:
            z = model.dynamics(z)
            x_pred = model.decoder(z)

        preds.append(torch.squeeze(x_pred).tolist())

    _, name = os.path.split(datapath)
    name = os.path.join(result_folder, name)
    DataFrame(preds).transpose().to_csv(name)

    return name
