"""The dataloader for VIPL-HR-V1 dataset.

Details for the VIPL-HR-V1 Dataset see https://vipl.ict.ac.cn/resources/databases/201811/t20181129_32716.html.

By using the VIPL-HR database, you are recommended to cite the following paper:
[1] Xuesong Niu, Shiguang Shan*, Hu Han, and Xilin Chen, "RhythmNet: End-to-end Heart Rate Estimation from Face via Spatial-temporal Representation",
 IEEE Transactions on Image Processing (T-IP), vol. 29, pp. 2409-2423, 2020.

[2] Xuesong Niu, Hu Han, Shiguang Shan, and Xilin Chen, “VIPL-HR: A Multi-modal Database for Pulse Estimation from Less-constrained Face Video”,
 Asian Conference on Computer Vision, 2018.
"""
import glob
import os

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class VIPLHRv1Loader(BaseLoader):
    """The data loader for the VIPL-HR-V1 dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an VIPL-HR-V1 dataloader.
        Args:
            data_path(str): path of a folder which stores raw video and bvp data.
            e.g. data_path should be "RawData" for below dataset structure:
            -----------------
                 RawData/
                 |   |-- p1/
                 |       |-- v1/
                 |          |-- source1
                 |              |-- video.avi
                 |              |-- wave.csv
                 |          |...
                 |          |-- source4
                 |              |-- video.avi
                 |              |-- wave.csv
                 |       |...
                 |       |-- v9/
                 |          |-- source1
                 |              |-- video.avi
                 |              |-- wave.csv
                 |          |...
                 |          |-- source4
                 |              |-- video.avi
                 |              |-- wave.csv
                 |...
                 |   |-- pn/
                 |       |-- v1/
                 |          |-- source1
                 |              |-- video.avi
                 |              |-- wave.csv
                 |          |...
                 |          |-- source4
                 |              |-- video.avi
                 |              |-- wave.csv
                 |       |...
                 |       |-- v9/
                 |          |-- source1
                 |              |-- video.avi
                 |              |-- wave.csv
                 |          |...
                 |          |-- source4
                 |              |-- video.avi
                 |              |-- wave.csv
            -----------------
            name(string): name of the dataloader.
            config_data(CfgNode): data settings(ref:config.py).
    """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For VIPL-HR-V1 dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1]
            sub_dirs = glob.glob(data_dir + os.sep + "*")

            # iterate over 9 recorded scenarios
            for sub_dir in sub_dirs:
                index_scenario = os.path.split(sub_dir)[-1]

                subsub_dirs = glob.glob(sub_dirs[0] + os.sep + "*")
                # iterate over 4 camera sources (source1: 960x720@25fps, source2&3: 1920x1080@30fps, source4: NIR camera 640x480@30fps)
                for subsub_dir in subsub_dirs:
                    index_camera_source = os.path.split(subsub_dir)[-1]

                    # take only HUAWEI P9 videos into Account
                    if index_camera_source == "source2":
                        dirs.append({"index": '-'.join([subject, index_scenario, index_camera_source]),
                                     "path": subsub_dir})  # ,
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

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Preprocesses the raw data."""

        # Read Video Frames
        file_num = len(data_dirs)
        for i in range(file_num):
            frames = self.read_video(
                os.path.join(
                    data_dirs[i]["path"],
                    "video.avi"))

            # Read Labels
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(
                    os.path.join(
                        data_dirs[i]["path"],
                        "wave.csv"))

            target_length = frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            self.preprocessed_data_len += self.save(frames_clips, bvps_clips, data_dirs[i]["index"])

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
            bvp = [float(x) for x in str1[1:-1]]
        return np.asarray(bvp)
