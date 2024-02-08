import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing

# Processed dataset path, need to be updated to specified path in neural_methods/trainer/TscanTrainer.py
# where the numpy arrays of time, GT BVP and predicted BVP get saved
video_file = "/data/rPPG_dataset/processed_dataset"


ground_truth = np.load(f'{video_file}/labels_numpy.npy')
BVP = np.load(f'{video_file}/pred_ppg_test_numpy.npy')
time = np.arange(BVP.shape[0]) / 30

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
ground_truth = scaler.fit_transform(ground_truth.reshape(-1, 1))  # [10:]
BVP = scaler.fit_transform(BVP.reshape(-1, 1))  # [10:]

font_size = 18
# set the font to Charter
font = {'family': 'serif', 'serif': ['Charter'], 'size': font_size}
plt.rc('font', **font)
# plt.rc('xtick', labelsize=font_size)
# plt.rc('ytick', labelsize=font_size)

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


color_gt = '#1f77b4'
color_rppg = '#ff7f0e'

fig, ax1 = plt.subplots()
ax1.plot(time, ground_truth, label='GT BVP', color=color_gt)
ax1.set_xlabel("Time (s)", fontsize=14)
ax1.set_ylabel("Referenz-BVP", fontsize=14, color=color_gt)
ax1.tick_params('y', labelsize=11, colors=color_gt)
ax1.tick_params(axis='x', labelsize=11)

ax2 = ax1.twinx()
ax2.plot(time, BVP, label='rPPG BVP', color=color_rppg)
ax2.set_xlabel("Time (s)", fontsize=14)
ax2.set_ylabel("rPPG-BVP", fontsize=14, color=color_rppg)
ax2.tick_params('y', labelsize=11, colors=color_rppg)

fig.tight_layout()

plt.grid(True)
plt.show()
