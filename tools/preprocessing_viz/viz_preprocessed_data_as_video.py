import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import cv2

root = tk.Tk()
root.withdraw()

# Specify path to preprocessed dataset folder
preprocessed_dataset_path = "/data/rPPG_dataset/mat_dataset"          # Raw dataset path, need to be updated

# visualize RGB frames, diff normalized frames, and ground truth (label) after preprocessing
def load_frames():
    input_file = filedialog.askopenfilename(title="Select a file", filetypes=(("numpy files","*.npy"), ("all files","*.*")), initialdir=preprocessed_dataset_path)
    input_data = np.load(os.path.join(preprocessed_dataset_path, input_file))
    print(f'The shape of the loaded chunk is {np.shape(input_data)}.')

    if input_data.shape[-1] == 6:
        rgb_frames = input_data[..., 3:]
        diff_normalized_frames = input_data[..., :3]

        # Create a side-by-side visualization
        frames = np.concatenate((rgb_frames, diff_normalized_frames), axis=2)
    elif input_data.shape[-1] == 3:
        # Use RGB frames directly for visualization
        frames = input_data
    else:
        raise ValueError("Invalid input_data shape. Expected shape (..., 3) or (..., 6).")

    return frames.astype(np.uint8)  # frames.astype(np.float32)

fps = 30

frames = load_frames()

for i, frame in enumerate(frames):
    frame = cv2.resize(frame, (144, 144))
    cv2.imshow('ROI frames', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
