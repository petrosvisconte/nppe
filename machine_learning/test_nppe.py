#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time


random.seed(1)
tf.random.set_seed(1)
np.random.seed(1)

# Replace 'train.csv' and 'test.csv' with the actual paths to your CSV files
train_data_path = '/home/pierre/projects/senior_project/nppe/data/datasets/swissrole_10000_0.csv'
test_data_path = '/home/pierre/projects/senior_project/nppe/data/datasets/swissroleh_test.csv'
# Replace 'train.csv' and 'test.csv' with the actual paths to your CSV files
low_dim_path = '/home/pierre/projects/senior_project/nppe/data/datasets/low_dim.csv'
low_dim_test_path = '/home/pierre/projects/senior_project/nppe/data/datasets/low_dim_test.csv'

# Load the data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
low_dim = pd.read_csv(low_dim_path)
low_dim_test = pd.read_csv(low_dim_test_path)

# Split the data into features and labels
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
X_train = low_dim.iloc[:, :-1]
X_test = low_dim_test.iloc[:, :-1]

# Assuming 'color_data' is the name of the column containing the Swiss roll color data
color_train = train_data.iloc[:, -1]
color_test = test_data.iloc[:, -1]

# Define the colormap
cmap = plt.cm.Spectral

# Normalize the color data to be between 0 and 1
norm = mcolors.Normalize(vmin=color_train.min(), vmax=color_train.max())

# Apply the colormap to the normalized color data to get the colors
y_train = norm(color_train)
#y_train = y_train[0]
y_test = norm(color_test)
print(y_train)


# Define more specific bins for the color data
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Define labels for each bin
color_labels = ['Red', 'Red-Orange', 'Orange', 'Yellow-Orange', 'Yellow', 'Yellow-Green', 'Green', 'Green-Blue', 'Blue', 'Blue-Violet']

y_train_labels = []
for i in range(len(y_train)):
    if 0 <= y_train[i] < 0.1:
        y_train_labels.append('Red')
    elif 0.1 <= y_train[i] < 0.2:
        y_train_labels.append('Red-Orange')
    elif 0.2 <= y_train[i] < 0.3:
        y_train_labels.append('Orange')
    elif 0.3 <= y_train[i] < 0.4:
        y_train_labels.append('Yellow-Orange')
    elif 0.4 <= y_train[i] < 0.5:
        y_train_labels.append('Yellow')
    elif 0.5 <= y_train[i] < 0.6:
        y_train_labels.append('Yellow-Green')
    elif 0.6 <= y_train[i] < 0.7:
        y_train_labels.append('Green')
    elif 0.7 <= y_train[i] < 0.8:
        y_train_labels.append('Green-Blue')
    elif 0.8 <= y_train[i] < 0.9:
        y_train_labels.append('Blue')
    elif 0.9 <= y_train[i] <= 1.0:
        y_train_labels.append('Blue-Violet')
for i in range(len(y_train)):
    if y_train_labels[i] == 'Red':
        y_train[i] = 0.05
    elif y_train_labels[i] == 'Red-Orange':
        y_train[i] = 0.15
    elif y_train_labels[i] == 'Orange':
        y_train[i] = 0.25
    elif y_train_labels[i] == 'Yellow-Orange':
        y_train[i] = 0.35
    elif y_train_labels[i] == 'Yellow':
        y_train[i] = 0.45
    elif y_train_labels[i] == 'Yellow-Green':
        y_train[i] = 0.55
    elif y_train_labels[i] == 'Green':
        y_train[i] = 0.65
    elif y_train_labels[i] == 'Green-Blue':
        y_train[i] = 0.75
    elif y_train_labels[i] == 'Blue':
        y_train[i] = 0.85
    elif y_train_labels[i] == 'Blue-Violet':
        y_train[i] = 0.95
y_test_labels = []
for i in range(len(y_test)):
    if 0 <= y_test[i] < 0.1:
        y_test_labels.append('Red')
    elif 0.1 <= y_test[i] < 0.2:
        y_test_labels.append('Red-Orange')
    elif 0.2 <= y_test[i] < 0.3:
        y_test_labels.append('Orange')
    elif 0.3 <= y_test[i] < 0.4:
        y_test_labels.append('Yellow-Orange')
    elif 0.4 <= y_test[i] < 0.5:
        y_test_labels.append('Yellow')
    elif 0.5 <= y_test[i] < 0.6:
        y_test_labels.append('Yellow-Green')
    elif 0.6 <= y_test[i] < 0.7:
        y_test_labels.append('Green')
    elif 0.7 <= y_test[i] < 0.8:
        y_test_labels.append('Green-Blue')
    elif 0.8 <= y_test[i] < 0.9:
        y_test_labels.append('Blue')
    elif 0.9 <= y_test[i] <= 1.0:
        y_test_labels.append('Blue-Violet')
    else:
        y_test_labels.append('Unknown')
for i in range(len(y_test)):
    if y_test_labels[i] == 'Red':
        y_test[i] = 0.05
    elif y_test_labels[i] == 'Red-Orange':
        y_test[i] = 0.15
    elif y_test_labels[i] == 'Orange':
        y_test[i] = 0.25
    elif y_test_labels[i] == 'Yellow-Orange':
        y_test[i] = 0.35
    elif y_test_labels[i] == 'Yellow':
        y_test[i] = 0.45
    elif y_test_labels[i] == 'Yellow-Green':
        y_test[i] = 0.55
    elif y_test_labels[i] == 'Green':
        y_test[i] = 0.65
    elif y_test_labels[i] == 'Green-Blue':
        y_test[i] = 0.75
    elif y_test_labels[i] == 'Blue':
        y_test[i] = 0.85
    elif y_test_labels[i] == 'Blue-Violet':
        y_test[i] = 0.95

# # Extract sr_points and sr_color
# n = 3
# sr_points = X_train.values
# sr_color = color_train
# # Plot the data
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, cmap="Spectral")
# ax.set_title("Training set - 2000 observations - noise=0", y=0.9)
# ax.view_init(azim=-66, elev=12)
# plt.show()
# # Extract sr_points and sr_color
# n = 3
# sr_points = X_train.values
# sr_color = y_train
# # Plot the data
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, cmap="Spectral")
# ax.set_title("Training set - 2000 observations - noise=0", y=0.9)
# ax.view_init(azim=-66, elev=12)
# plt.show()



# # Extract sr_points and sr_color
# n = 2
# sr_points = X_train.values
# sr_color = color_train
# # Plot the data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(sr_points[:, 0], sr_points[:, 1], c=sr_color, cmap="Spectral")
# ax.set_title("Training set - low dimension", fontsize=16)
# #ax.text(-2, 0, 25, f"Number of samples: {10}", fontsize=9)
# #ax.text(-2, 0, 28, f"Noise: {0}", fontsize=9)
# plt.show()
# # Extract sr_points and sr_color
# n = 2
# sr_points = X_train.values
# sr_color = y_train
# # Plot the data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(sr_points[:, 0], sr_points[:, 1], c=sr_color, cmap="Spectral")
# ax.set_title("Training set - low dimension", fontsize=16)
# #ax.text(-2, 0, 25, f"Number of samples: {10}", fontsize=9)
# #ax.text(-2, 0, 28, f"Noise: {0}", fontsize=9)
# plt.show()
# # Extract sr_points and sr_color
# n = 2
# sr_points = X_test.values
# sr_color = color_test
# # Plot the data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(sr_points[:, 0], sr_points[:, 1], c=sr_color, cmap="Spectral")
# ax.set_title("Training set - low dimension", fontsize=16)
# #ax.text(-2, 0, 25, f"Number of samples: {10}", fontsize=9)
# #ax.text(-2, 0, 28, f"Noise: {0}", fontsize=9)
# plt.show()
# # Extract sr_points and sr_color
# n = 2
# sr_points = X_test.values
# sr_color = y_test
# # Plot the data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(sr_points[:, 0], sr_points[:, 1], c=sr_color, cmap="Spectral")
# ax.set_title("Training set - low dimension", fontsize=16)
# #ax.text(-2, 0, 25, f"Number of samples: {10}", fontsize=9)
# #ax.text(-2, 0, 28, f"Noise: {0}", fontsize=9)
# plt.show()




# Convert y_train_labels to a one-hot encoded numpy array
lb = LabelBinarizer()
y_train_labels = lb.fit_transform(y_train_labels)
num_classes = len(color_labels)  # Number of unique labels

# Define the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # For multi-class classification
])
start_time = time.time()
# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_labels, epochs=200, batch_size=30)

# Convert y_test_labels to a one-hot encoded numpy array
#y_test_labels_encoded = lb.transform(y_test_labels)
# Make predictions on the test data
predictions = model.predict(X_test)
# Convert the predictions back to labels
predicted_labels = lb.inverse_transform(predictions)
# Calculate the accuracy
accuracy = np.mean(predicted_labels == y_test_labels)
print(f'Accuracy: {accuracy * 100}%')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")



# TODO: convert prediction_labels to float values and then plot them to compare with the actual values