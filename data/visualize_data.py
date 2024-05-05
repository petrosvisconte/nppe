#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys


def main(args):
    # Read data from CSV file
    data = pd.read_csv(args.h)
    # Extract sr_points and sr_color
    n = 3
    sr_points = data.iloc[:, :n].values
    sr_color = data.iloc[:, n].values
    
    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, cmap="Spectral")
    if args.h:
        ax.set_title("Training set - 2000 observations - noise=0", y=0.9)
    else:
        ax.set_title("Training set - 2000 observations - noise=0", y=0.9)
    ax.view_init(azim=-66, elev=12)
    plt.show()


    # Read data from CSV file
    data = pd.read_csv(args.l)
    # Extract sr_points and sr_color
    n = 2
    sr_points = data.iloc[:, :n].values
    
    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(sr_points[:, 0], sr_points[:, 1], c=sr_color, cmap="Spectral")
    if args.h:
        ax.set_title("Training set - low dimension", fontsize=16)
    else:
        ax.set_title("Training set - low dimension", fontsize=16)
    #ax.text(-2, 0, 25, f"Number of samples: {10}", fontsize=9)
    #ax.text(-2, 0, 28, f"Noise: {0}", fontsize=9)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=str, help='File name of high dimensional data')
    parser.add_argument('--l', type=str, default="datasets/low_dim.csv", help='File name of low dimensional data')
    args = parser.parse_args()

    if not any(vars(args).values()):
        print("No file name arguments provided. Exiting.")
        sys.exit()

    main(args)