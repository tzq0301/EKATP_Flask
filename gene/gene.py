import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def gene(data_path: str):
    X = pd.read_csv(data_path)
    # print(X.shape)  # 84维 22步 （84，22）
    # print(len(X))
    X = X.T
    X = np.array(X)
    # print(X.shape)
    # print(type(X))

    xmin = np.min(X)
    xptp = np.ptp(X)
    # scale
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    # print(X.shape)

    # split into train and test set
    X_train = X[0:15]
    X_test = X[15:]

    # ******************************************************************************
    # Return train and test set
    # ******************************************************************************
    return X_train, X_test, len(X), 1, xmin, xptp


def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1], 1)

    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    else:
        return "dimenional error"


def example(model, datapath: str, step: int = 6):
    # ******************************************************************************
    # load data
    # ******************************************************************************
    Xtrain, Xtest, m, n, xmin, xptp = gene(datapath)

    # ******************************************************************************
    # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
    # ******************************************************************************
    # Xtrain = add_channels(Xtrain)
    Xtest = add_channels(Xtest)
    # transfer to tensor
    # Xtrain = torch.from_numpy(Xtrain).float().contiguous()
    Xtest = torch.from_numpy(Xtest).float().contiguous()

    # ******************************************************************************
    # Prediction
    # ******************************************************************************
    Xinput, Xtarget = Xtest[:-1], Xtest[1:]

    init = Xinput[0].float()

    preds = []
    targets = []

    z = model.encoder(init)

    for j in range(step):
        if isinstance(z, tuple):
            z = model.dynamics(*z)
            x_pred = model.decoder(z[0])
        else:
            z = model.dynamics(z)
            x_pred = model.decoder(z)

        preds.append(torch.squeeze(x_pred))
        targets.append(torch.squeeze(Xtarget[j].data))

    plt.figure(figsize=(13, 64))
    x = np.arange(0, len(preds))
    for i in range(m):
        preds_first_point = [tensor[i].item() for tensor in preds]
        targets_first_point = [tensor[i].item() for tensor in targets]

        plt.subplot(m, 4, i + 1)
        # plt.title(f"Dimension {i + 1}")
        plt.ylim(-1, 1)
        # plt.axis('off')
        plt.plot(x, preds_first_point, color='red', label='pred')
        plt.plot(x, targets_first_point, color='blue', label='target')
        plt.legend()

    filename: str = 'results/example.png'
    filepath: str = str(pathlib.Path(f'./static/{filename}').resolve())

    plt.savefig(filepath, bbox_inches='tight')

    return filename


def predict(model, datapath: str, step: int = 6):
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

        preds.append(torch.squeeze(x_pred))

    plt.figure(figsize=(13, 13 * len(X) // 4))
    x = np.arange(0, len(preds))
    for i in range(len(X)):
        preds_first_point = [tensor[i].item() for tensor in preds]

        plt.subplot(len(X), 4, i + 1)
        plt.ylim(-1, 1)
        plt.plot(x, preds_first_point, color='red', label='pred')
        plt.legend()

    filename: str = 'results/example.png'
    filepath: str = str(pathlib.Path(f'./static/{filename}').resolve())

    plt.savefig(filepath, bbox_inches='tight')

    return filename
