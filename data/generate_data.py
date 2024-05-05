#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from keras.datasets import mnist


# Generate the swiss roll dataset with s samples and n noise
def swissRoll(s, n, h):
    # Generate the data
    sr_points, sr_color = datasets.make_swiss_roll(n_samples=s, noise=n, hole=h, random_state=0)
    
    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, cmap="Spectral")
    if h==True:
        ax.set_title("Swiss Hole Dataset")
    else:
        ax.set_title("Swiss Roll Dataset")
    ax.view_init(azim=-66, elev=12)
    ax.text(-2, 0, 25, f"Number of samples: {s}", fontsize=9)
    ax.text(-2, 0, 28, f"Noise: {n}", fontsize=9)
    plt.show()
    
    # Write data to .csv
    if h:
        np.savetxt("datasets/swissroleh_"+str(s)+"_"+str(n)+".csv", np.column_stack((sr_points,sr_color)), delimiter=",")
    else:
        np.savetxt("datasets/swissrole_"+str(s)+"_"+str(n)+".csv", np.column_stack((sr_points,sr_color)), delimiter=",")


def main():
    # create swissroll datasets
    swissRoll(20, 0, False)
    swissRoll(2000,0.5, False)
    # create swissroll datasets with a hole
    swissRoll(2000, 0, True)
    swissRoll(2000,0.5, True)

    # Generate a new testing set of 500 points
    sr_points, sr_color = datasets.make_swiss_roll(n_samples=500, noise=0.5, hole=False, random_state=1)  # Use a different random state
    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, cmap="Spectral")
    ax.set_title("Swiss Hole Dataset")
    ax.view_init(azim=-66, elev=12)
    ax.text(-2, 0, 25, f"Number of observations: {500}", fontsize=9)
    ax.text(-2, 0, 28, f"Noise: {0.5}", fontsize=9)
    plt.show()
    # Write data to .csv
    np.savetxt("datasets/swissroleh_test.csv", np.column_stack((sr_points,sr_color)), delimiter=",")

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # create a DataFrame from the training set
    train_df = pd.DataFrame(train_X.reshape(-1, 784))
    train_df['label'] = train_y

    # create a DataFrame from the testing set
    test_df = pd.DataFrame(test_X.reshape(-1, 784))
    test_df['label'] = test_y

    # write the DataFrames to csv files
    train_df.to_csv('datasets/mnist_train.csv', sep='\t', header=False, index=False)
    test_df.to_csv('datasets/mnist_test.csv', sep='\t', header=False, index=False)
    print(len(train_df.columns))



    # create a list of copies of the train dataframe
    copies = [train_df] * 20

    # concatenate the copies along the row axis
    train_df_multiplied = pd.concat(copies, axis=0)

    # reset the index of the concatenated dataframe
    train_df_multiplied.reset_index(drop=True, inplace=True)
    train_df_multiplied.to_csv('datasets/mnist_train_ext_2.csv', sep='\t', header=False, index=False)
    print(len(train_df_multiplied))


if __name__ == "__main__":
    main()