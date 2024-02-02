import os

import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

# Specify path to dataset folder
dataset_path = 'i:/Datasets/KISMED/data/'


# Read the CSV file into a pandas DataFrame
input_file = filedialog.askopenfilename(title="Select a file",
                                        filetypes=(("bvp files", "*.csv"),),
                                        initialdir=dataset_path)
input_data = os.path.join(dataset_path, input_file)
data = pd.read_csv(input_data)

# Convert the 'timestamp' column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', utc=True)
data['timestamp'] = data['timestamp'].dt.tz_convert('Europe/Berlin')

plot_length = -1

# Plotting the time series data
plt.figure(figsize=(10, 6))
plt.plot(data['timestamp'][:plot_length], data['bvp'][:plot_length], linestyle='-')
plt.title('Blood Volume Pulse Time Series')
plt.xlabel('Timestamp')
plt.ylabel('BVP')
plt.grid(True)
plt.show()