import matplotlib.pyplot as plt
import numpy as np

# Processed dataset path, need to be updated to specified path in unsupervised_methods/unsupervised_predictor.py,
# where the numpy arrays of time, GT BVP and predicted BVP get saved
video_file = "/data/rPPG_dataset/processed_dataset"

_ = 11
idx = 0
i = 0
method_name = "POS"

time = np.load(f'{video_file}/time_{_}_{idx}_{i}.npy')
ground_truth = np.load(f'{video_file}/ground_truth_{_}_{idx}_{i}.npy')
BVP = np.load(f'{video_file}/{method_name}_BVP_{_}_{idx}_{i}.npy')

font_size = 18
# set the font to Charter
font = {'family': 'serif', 'serif': ['Charter'], 'size': font_size}
plt.rc('font', **font)
# plt.rc('xtick', labelsize=font_size)
# plt.rc('ytick', labelsize=font_size)

SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

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
ax1.set_ylabel("Ground Truth BVP Normalized", fontsize=14, color=color_gt)
ax1.tick_params('y', labelsize=11, colors=color_gt)
ax1.tick_params(axis='x', labelsize=11)

ax2 = ax1.twinx()
ax2.plot(time, BVP, label=method_name + ' BVP', color=color_rppg)
ax2.set_xlabel("Time (s)", fontsize=14)
ax2.set_ylabel(method_name + " BVP Normalized", fontsize=14, color=color_rppg)
ax2.tick_params('y', labelsize=11, colors=color_rppg)

fig.tight_layout()

plt.grid(True)
plt.show()
