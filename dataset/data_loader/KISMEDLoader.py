"""

The dataloader for KISMED dataset.

"""
import glob
import os

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class KISMEDLoader(BaseLoader):
    """The data loader for the KISMED dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an KISMED dataloader.
        Args:
            data_path(str): path of a folder which stores raw video and bvp data.
            e.g. data_path should be "RawData" for below dataset structure:
            -----------------
                 RawData/
                 |   |-- p001/
                 |       |-- v01/
                 |          |-- video_RAW_RGBA.avi
                 |          |-- BVP.csv
                 |       |...
                 |       |-- v12/
                 |          |-- video_RAW_RGBA.avi
                 |          |-- BVP.csv
                 |...
                 |   |-- p010/
                 |       |-- v01/
                 |          |-- video_RAW_RGBA.avi
                 |          |-- BVP.csv
                 |       |...
                 |       |-- v12/
                 |          |-- video_RAW_RGBA.avi
                 |          |-- BVP.csv
            -----------------
            name(string): name of the dataloader.
            config_data(CfgNode): data settings(ref:config.py).
    """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For KISMED dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1]

            sub_dirs = glob.glob(data_dir + os.sep + "*")

            # iterate over 12 recorded scenarios
            for sub_dir in sub_dirs:
                index_scenario = os.path.split(sub_dir)[-1]

                # use only "rotation" rPPG scenario: v11
                # v01: sationary ceiling-illumination, v02: stationary natural-illumination, 
                # v03: altering illumination, v04:varying side-illumination (ceil on), 
                # v05: varying side_illumination (ceil off), v06: varying camera-distance,
                # v07: squats, v08: face wear, v09: natural behaviour, v10: head translation,
                # v11: head rotation, v12: head rotation and translation
                if index_scenario not in self.config_data.SCENARIOS: #["v01","v02","v03","v04","v05","v06","v07","v08","v09","v10","v11","v12"]:
                    continue
                else:
                    dirs.append({"index": '-'.join([subject, index_scenario]),
                                 "path": sub_dir})  # ,
                                 # "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocesses the raw data."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(os.path.join(data_dirs[i]["path"], "video_RAW_RGBA.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(os.path.join(data_dirs[i]["path"], "BVP.csv"))

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, saved_filename)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x.split(",")[1]) for x in str1[1:-1]]
        return np.asarray(bvp)
