"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.

"""
import csv
import glob
import os
import re
from math import ceil

import scipy.spatial.distance
from scipy import signal
from scipy import sparse
from unsupervised_methods.methods import POS_WANG
from unsupervised_methods import utils
import math
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

import mediapipe as mp
from scipy.signal import iirfilter, filtfilt
from roi_segmentation import DEFINITION_FACEMASK
from roi_segmentation.helper_code import calc_triangle_centroid_coordinates, check_acceptance, extract_mask_outside_roi, \
    apply_bounding_box, skin_segmentation, generate_face_mask
from roi_segmentation.helper_code import segment_roi, apply_convex_hull, mask_eyes_out, calc_centroid_between_roi, count_pixel_area, \
    interpolate_surface_normal_angles_scipy, get_bounding_box_coordinates, get_bounding_box_coordinates_mesh_points, get_bounding_box_coordinates_filtered
from roi_segmentation.surface_normal_vector.helper_functions import get_triangle_coords, calculate_surface_normal_angle, get_triangle_indices_from_angle

import time
from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface




class BaseLoader(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument(
            "--cached_path", default=None, type=str)
        parser.add_argument(
            "--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """Inits dataloader with lists of files.

        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = list()
        self.labels = list()
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data

        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)
        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        else:
            if not os.path.exists(self.cached_path):
                print('CACHED_PATH:', self.cached_path)
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')

            self.load_preprocessed_data()
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        label = np.float32(label)
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments,
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

    def get_raw_data(self, raw_data_path):
        """Returns raw data directories under the path.

        Args:
            raw_data_path(str): a list of video_files.
        """
        raise Exception("'get_raw_data' Not Implemented")

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        raise Exception("'split_raw_data' Not Implemented")

    def read_npy_video(self, video_file):
        """Reads a video file in the numpy format (.npy), returns frames(T,H,W,3)"""
        frames = np.load(video_file[0])
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range of values! '\
            + f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
        return np.asarray(processed_frames)

    def generate_pos_psuedo_labels(self, frames, fs=30):
        """Generated POS-based PPG Psuedo Labels For Training

        Args:
            frames(List[array]): a video frames.
            fs(int or float): Sampling rate of video
        Returns:
            env_norm_bvp: Hilbert envlope normalized POS PPG signal, filtered are HR frequency
        """

        # generate POS PPG signal
        WinSec = 1.6
        RGB = POS_WANG._process_video(frames)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fs)

        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])

        bvp = H
        bvp = utils.detrend(np.mat(bvp).H, 100)
        bvp = np.asarray(np.transpose(bvp))[0]

        # filter POS PPG w/ 2nd order butterworth filter (around HR freq)
        # min freq of 0.7Hz was experimentally found to work better than 0.75Hz
        min_freq = 0.70
        max_freq = 3
        b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
        pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))

        # apply hilbert normalization to normalize PPG amplitude
        analytic_signal = signal.hilbert(pos_bvp)
        amplitude_envelope = np.abs(analytic_signal) # derive envelope signal
        env_norm_bvp = pos_bvp/amplitude_envelope # normalize by env

        return np.array(env_norm_bvp) # return POS psuedo labels

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all the raw data based on split.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)  # partition dataset
        # send data directories to be processed
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess)
        self.build_file_list(file_list_dict)  # build file list
        self.load_preprocessed_data()  # load all data and corresponding labels (sorted for consistency)
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocess(self, frames, bvps, config_preprocess, saved_filename):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        Returns:
            frame_clips(np.array): processed video data by frames
            bvps_clips(np.array): processed bvp (ppg) labels by frames
        """

        # perform ROI segmentation based on defined threshold for reflectance angles
        frames = self.roi_segmentation_for_video(frames, saved_filename,
                                       config_preprocess.ROI_SEGMENTATION.DO_SEGMENTATION,
                                       config_preprocess.ROI_SEGMENTATION.THRESHOLD,
                                       config_preprocess.ROI_SEGMENTATION.ROI_MODE,
                                       config_preprocess.ROI_SEGMENTATION.USE_CONVEX_HULL,
                                       config_preprocess.ROI_SEGMENTATION.CONSTRAIN_ROI,
                                       config_preprocess.ROI_SEGMENTATION.USE_OUTSIDE_ROI)

        if "Standardized_ROI_segmentation" in config_preprocess.DATA_TYPE:
            # perform ROI segmentation to extract the whole face from video
            frames_whole_face = self.roi_segmentation_for_video(frames, saved_filename,
                                                         use_roi_segmentation=True,
                                                         threshold=90,
                                                         roi_mode="optimal_roi",
                                                         use_convex_hull=False,
                                                         constrain_roi=False,
                                                         use_outside_roi=False,
                                                         apply_heatmap=False)


            # perform ROI segmentation to extract forehead and cheeks ROI
            frames_roi = self.roi_segmentation_for_video(frames, saved_filename,
                                                         use_roi_segmentation=True,
                                                         threshold=90,
                                                         roi_mode="optimal_roi",
                                                         use_convex_hull=True,
                                                         constrain_roi=True,
                                                         use_outside_roi=False,
                                                         apply_heatmap=False)

        if "Standardized_ROI_segmentation_heatmap" in config_preprocess.DATA_TYPE:
            # perform ROI segmentation to extract the whole face from video
            frames_whole_face = self.roi_segmentation_for_video(frames, saved_filename,
                                                         use_roi_segmentation=True,
                                                         threshold=90,
                                                         roi_mode="optimal_roi",
                                                         use_convex_hull=False,
                                                         constrain_roi=False,
                                                         use_outside_roi=False,
                                                         apply_heatmap=False)


            # perform ROI segmentation to extract forehead and cheeks ROI
            frames_roi = self.roi_segmentation_for_video(frames, saved_filename,
                                                         use_roi_segmentation=True,
                                                         threshold=90,
                                                         roi_mode="optimal_roi",
                                                         use_convex_hull=False,
                                                         constrain_roi=True,
                                                         use_outside_roi=False,
                                                         apply_heatmap=True)

        if "Standardized_FACE_segmentation" in config_preprocess.DATA_TYPE:
            # perform ROI segmentation to extract the whole face from video
            frames_whole_face = self.roi_segmentation_for_video(frames, saved_filename,
                                                         use_roi_segmentation=True,
                                                         threshold=90,
                                                         roi_mode="optimal_roi",
                                                         use_convex_hull=False,
                                                         constrain_roi=False,
                                                         use_outside_roi=False,
                                                         apply_heatmap=False)

        # resize frames and crop for face region
        frames = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)
        # Check data transformation type
        data = list()  # Video data
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                if ("Standardized_ROI_segmentation" in config_preprocess.DATA_TYPE
                        or "Standardized_FACE_segmentation" in config_preprocess.DATA_TYPE
                        or "Standardized_ROI_segmentation_heatmap" in config_preprocess.DATA_TYPE):
                    f_wface = frames_whole_face.copy()
                    data.append(BaseLoader.diff_normalize_data(f_wface))
                else:
                    data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            elif data_type == "Standardized_ROI_segmentation":
                f_roi = frames_roi.copy()
                data.append(BaseLoader.standardized_data(f_roi))
            elif data_type == "Standardized_ROI_segmentation_heatmap":
                f_roi_heatmap = frames_roi.copy()
                data.append(BaseLoader.standardized_data(f_roi_heatmap))
            elif data_type == "Standardized_FACE_segmentation":
                data.append(BaseLoader.standardized_data(f_wface))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, bvps_clips = self.chunk(
                data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    def calculate_roi(self, image, skin_segmentation_mask, threshold=90, roi_mode="optimal_roi", constrain_roi=True, use_convex_hull=True, use_outside_roi=False):
        """
         Calculates and extracts the region of interest (ROI) from an input image based on facial landmarks and their angles with respect to camera and surface normals.
         It uses the results from facial landmark detection to identify and extract triangles from the face mesh that fall below the specified angle threshold.
         The extracted triangles are returned as a list of sets of coordinates, which define the adaptive ROI.
         Additionally the function returns the binary mask images of the optimal ROI and the ROI outside of the optimal region.

        :param image: The input image where the facial landmarks were detected in and the adaptive ROI is to be calculated for.
        :param skin_segmentation_mask: The previously computed facial skin segmentation mask. Must have the same dimensions as the image.
        :param threshold:(int, optional, default=90): The angle threshold in degrees. Triangles with angles below this threshold will be included in the adaptive ROI.
        :param constrain_roi:(bool, optional, default=True): A flag indicating whether to constrain the adaptive ROI to a predefined set of optimal regions.
                                        If set to True, only triangles within the predefined regions will be considered.
        :param use_outside_roi: (bool, optional, default=False): If True, calculate ROIs outside of the constrained ROI.
        :return: tuple: A tuple containing the following elements:
                - mesh_points_threshold_roi_ (list): List of sets of coordinates defining triangles within the optimal ROI that meet the angle threshold criteria and,
                                                   if flag is set, are part of the optimal ROI.
                - mesh_points_outside_roi (list): List of coordinates defining triangles outside of the optimal ROI.
                - mask_threshold_roi (numpy.ndarray): A binary image mask indicating the optimal ROI.
                - mask_outside_roi (numpy.ndarray): A binary image mask indicating the area outside the optimal ROI.

        A list of sets of coordinates representing the triangles that meet the angle threshold criteria and, if flag is set, are part of the adaptive ROI.
        """
        img_h, img_w = image.shape[:2]

        mesh_points_forehead = []
        mesh_points_left_cheek = []
        mesh_points_right_cheek = []
        mask_optimal_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_forehead_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_left_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_right_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_outside_roi = np.zeros((img_h, img_w), dtype=np.uint8)

        mean_angle_forehead = []
        mean_angle_left_cheek = []
        mean_angle_right_cheek = []

        # define tesselation triangles contained in each roi
        if constrain_roi:
            forehead_roi = DEFINITION_FACEMASK.FOREHEAD_TESSELATION_LARGE
            left_cheek_roi = DEFINITION_FACEMASK.LEFT_CHEEK_TESSELATION_LARGE
            right_cheek_roi = DEFINITION_FACEMASK.RIGHT_CHEEK_TESSELATION_LARGE

        # Extract filtered landmark xyz-coordinates from the detected face in video
        landmark_coords_xyz = np.zeros((len(landmark_coords_xyz_history), 3))
        for index, face_landmarks in enumerate(landmark_coords_xyz_history):
            landmark_coords_xyz[index] = [face_landmarks[video_frame_count][0], face_landmarks[video_frame_count][1], face_landmarks[video_frame_count][2]]

        # initialization for surface angle interpolation for all face pixels
        x_min, x_max = int(landmark_coords_xyz[:, 0].min() * img_w), int(landmark_coords_xyz[:, 0].max() * img_w)
        y_min, y_max = int(landmark_coords_xyz[:, 1].min() * img_h), int(landmark_coords_xyz[:, 1].max() * img_h)
        xx, yy = np.meshgrid(np.arange(x_max - x_min), np.arange(y_max - y_min))

        # image pixel coordinates for which to interpolate surface normal angles, each row is [x, y], starting from [0, 0] -> [img_w, img_h]
        pixel_coordinates = np.column_stack((xx.ravel(), yy.ravel()))

        # [x, y] coordinates of the triangle centroids
        centroid_coordinates = np.zeros((len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION), 2), dtype=np.int32)
        # surface normal angles for each triangle centroid
        surface_normal_angles = np.zeros(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))

        # Calculate angles between camera and surface normal vectors for whole face mesh tessellation
        for index, triangle in enumerate(DEFINITION_FACEMASK.FACE_MESH_TESSELATION):
            # calculate reflectance angle in degree
            angle_degrees = calculate_surface_normal_angle(landmark_coords_xyz, triangle)

            # calculate centroid coordinates of each triangle
            triangle_centroid = np.mean(np.array([landmark_coords_xyz[i] for i in triangle]), axis=0)
            centroid_coordinates[index] = calc_triangle_centroid_coordinates(triangle_centroid, img_w, img_h, x_min, y_min)

            # for interpolation, the reflectance angle is calculated for each triangle and mapped to the triangles centroid
            surface_normal_angles[index] = angle_degrees

            # check acceptance of triangle to be below threshold and add it to the ROI mask
            if constrain_roi:
                # Extract the coordinates of the three landmarks of the triangle
                triangle_coords = get_triangle_coords(image, landmark_coords_xyz, triangle)

                if triangle in forehead_roi:
                    mesh_points_forehead.append(triangle_coords)  # necessary for bounding box
                    if check_acceptance(index, angle_degrees, angle_history, threshold):
                        cv2.fillConvexPoly(mask_forehead_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
                        mean_angle_forehead.append(angle_degrees)
                elif triangle in left_cheek_roi:
                    mesh_points_left_cheek.append(triangle_coords)  # necessary for bounding box
                    if check_acceptance(index, angle_degrees, angle_history, threshold):
                        cv2.fillConvexPoly(mask_left_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
                        mean_angle_left_cheek.append(angle_degrees)
                elif triangle in right_cheek_roi:
                    mesh_points_right_cheek.append(triangle_coords)  # necessary for bounding box
                    if check_acceptance(index, angle_degrees, angle_history, threshold):
                        cv2.fillConvexPoly(mask_right_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
                        mean_angle_right_cheek.append(angle_degrees)
            else:
                # Extract the coordinates of the three landmarks of the triangle
                triangle_coords = get_triangle_coords(image, landmark_coords_xyz, triangle)
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_outside_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))

        if use_convex_hull:
            mask_forehead_roi = apply_convex_hull(mask_forehead_roi)
            mask_left_cheek_roi = apply_convex_hull(mask_left_cheek_roi)
            mask_right_cheek_roi = apply_convex_hull(mask_right_cheek_roi)

        # calculate pixel area of optimal_roi
        if roi_mode == "optimal_roi":
            mask_optimal_roi = mask_forehead_roi + mask_left_cheek_roi + mask_right_cheek_roi
            mesh_points_bounding_box_ = mesh_points_forehead + mesh_points_left_cheek + mesh_points_right_cheek
        elif roi_mode == "forehead":
            mask_optimal_roi = mask_forehead_roi
            mesh_points_bounding_box_ = mesh_points_forehead
        elif roi_mode == "left_cheek":
            mask_optimal_roi = mask_left_cheek_roi
            mesh_points_bounding_box_ = mesh_points_left_cheek
        elif roi_mode == "right_cheek":
            mask_optimal_roi = mask_right_cheek_roi
            mesh_points_bounding_box_ = mesh_points_right_cheek
        else:
            raise Exception("No valid roi_mode selected. Valid roi_mode are: 'optimal_roi', 'forehead', 'left_cheek', 'right_cheek'.")

        if skin_segmentation_mask is not None:
            mask_optimal_roi = cv2.bitwise_and(mask_optimal_roi, mask_optimal_roi, mask=skin_segmentation_mask)

        if constrain_roi and use_outside_roi:
            interpolated_surface_normal_angles = interpolate_surface_normal_angles_scipy(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max)
            mask_eyes = mask_eyes_out(mask_outside_roi, landmark_coords_xyz)
            # extract smallest interpolation angles and create new mask only including pixels with the same amount as mask_optimal_roi
            mask_outside_roi = extract_mask_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, mask_eyes, x_min, y_min)

        # save pixel areas and mean angles of optimal ROIs in a csv file
        if constrain_roi and roi_mode == "optimal_roi":
            global csv_writer
            csv_writer.writerow([video_frame_count,
                                 count_pixel_area(mask_optimal_roi + mask_outside_roi),
                                 count_pixel_area(mask_optimal_roi),
                                 count_pixel_area(mask_outside_roi),
                                 count_pixel_area(mask_outside_roi) - count_pixel_area(mask_optimal_roi),
                                 count_pixel_area(mask_forehead_roi),
                                 count_pixel_area(mask_left_cheek_roi),
                                 count_pixel_area(mask_right_cheek_roi),
                                 round(np.average(mean_angle_forehead), 3),
                                 round(np.average(mean_angle_left_cheek), 3),
                                 round(np.average(mean_angle_right_cheek), 3)])

        return mesh_points_bounding_box_, mask_optimal_roi, mask_outside_roi

    def map_value_to_rgb(self, value):
        """
           Maps a given value within the range [0, 90] to an RGB tuple in the range [255, 255, 255] to [0, 0, 0].

           Parameters:
           - value (float): The input value to be mapped, should be within the range [0, 90].

           Returns:
           - tuple: An RGB tuple representing the mapped color, where each component is in the range [0, 255].
           """
        # Ensure the value is within the range [0, 90]
        value = max(0, min(90, value))

        # Convert the range from [0, 90] to [255, 0]
        mapped_value = 255 - int((value / 60) * 255)

        rgb_tuple = (mapped_value, mapped_value, mapped_value)

        return rgb_tuple, mapped_value


    def calculate_roi_heatmap(self, image, skin_segmentation_mask, threshold=90, roi_mode="optimal_roi", constrain_roi=True, use_convex_hull=True, use_outside_roi=False):
        """
         Calculates and extracts the region of interest (ROI) from an input image based on facial landmarks and their angles with respect to camera and surface normals.
         It uses the results from facial landmark detection to identify and extract triangles from the face mesh that fall below the specified angle threshold.
         The extracted triangles are returned as a list of sets of coordinates, which define the adaptive ROI.
         Additionally the function returns the binary mask images of the optimal ROI and the ROI outside of the optimal region.

        :param image: The input image where the facial landmarks were detected in and the adaptive ROI is to be calculated for.
        :param skin_segmentation_mask: The previously computed facial skin segmentation mask. Must have the same dimensions as the image.
        :param threshold:(int, optional, default=90): The angle threshold in degrees. Triangles with angles below this threshold will be included in the adaptive ROI.
        :param constrain_roi:(bool, optional, default=True): A flag indicating whether to constrain the adaptive ROI to a predefined set of optimal regions.
                                        If set to True, only triangles within the predefined regions will be considered.
        :param use_outside_roi: (bool, optional, default=False): If True, calculate ROIs outside of the constrained ROI.
        :return: tuple: A tuple containing the following elements:
                - mesh_points_threshold_roi_ (list): List of sets of coordinates defining triangles within the optimal ROI that meet the angle threshold criteria and,
                                                   if flag is set, are part of the optimal ROI.
                - mesh_points_outside_roi (list): List of coordinates defining triangles outside of the optimal ROI.
                - mask_threshold_roi (numpy.ndarray): A binary image mask indicating the optimal ROI.
                - mask_outside_roi (numpy.ndarray): A binary image mask indicating the area outside the optimal ROI.

        A list of sets of coordinates representing the triangles that meet the angle threshold criteria and, if flag is set, are part of the adaptive ROI.
        """
        img_h, img_w = image.shape[:2]

        mesh_points_forehead = []
        mesh_points_left_cheek = []
        mesh_points_right_cheek = []
        mask_optimal_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_forehead_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_left_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_right_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_outside_roi = np.zeros((img_h, img_w), dtype=np.uint8)

        mean_angle_forehead = []
        mean_angle_left_cheek = []
        mean_angle_right_cheek = []

        # define tesselation triangles contained in each roi
        if constrain_roi:
            forehead_roi = DEFINITION_FACEMASK.FOREHEAD_TESSELATION_LARGE
            left_cheek_roi = DEFINITION_FACEMASK.LEFT_CHEEK_TESSELATION_LARGE
            right_cheek_roi = DEFINITION_FACEMASK.RIGHT_CHEEK_TESSELATION_LARGE

        # Extract filtered landmark xyz-coordinates from the detected face in video
        landmark_coords_xyz = np.zeros((len(landmark_coords_xyz_history), 3))
        for index, face_landmarks in enumerate(landmark_coords_xyz_history):
            landmark_coords_xyz[index] = [face_landmarks[video_frame_count][0], face_landmarks[video_frame_count][1], face_landmarks[video_frame_count][2]]

        # initialization for surface angle interpolation for all face pixels
        x_min, x_max = int(landmark_coords_xyz[:, 0].min() * img_w), int(landmark_coords_xyz[:, 0].max() * img_w)
        y_min, y_max = int(landmark_coords_xyz[:, 1].min() * img_h), int(landmark_coords_xyz[:, 1].max() * img_h)
        xx, yy = np.meshgrid(np.arange(x_max - x_min), np.arange(y_max - y_min))

        # image pixel coordinates for which to interpolate surface normal angles, each row is [x, y], starting from [0, 0] -> [img_w, img_h]
        pixel_coordinates = np.column_stack((xx.ravel(), yy.ravel()))

        # [x, y] coordinates of the triangle centroids
        centroid_coordinates = np.zeros((len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION), 2), dtype=np.int32)
        # surface normal angles for each triangle centroid
        surface_normal_angles = np.zeros(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))

        # Calculate angles between camera and surface normal vectors for whole face mesh tessellation
        for index, triangle in enumerate(DEFINITION_FACEMASK.FACE_MESH_TESSELATION):
            # calculate reflectance angle in degree
            angle_degrees = calculate_surface_normal_angle(landmark_coords_xyz, triangle)

            # calculate centroid coordinates of each triangle
            triangle_centroid = np.mean(np.array([landmark_coords_xyz[i] for i in triangle]), axis=0)
            centroid_coordinates[index] = calc_triangle_centroid_coordinates(triangle_centroid, img_w, img_h, x_min, y_min)

            # for interpolation, the reflectance angle is calculated for each triangle and mapped to the triangles centroid
            surface_normal_angles[index] = angle_degrees

            # check acceptance of triangle to be below threshold and add it to the ROI mask
            if constrain_roi:
                # Extract the coordinates of the three landmarks of the triangle
                triangle_coords = get_triangle_coords(image, landmark_coords_xyz, triangle)

                _, mapped_grayscale_value = self.map_value_to_rgb(angle_degrees)

                if triangle in forehead_roi:
                    mesh_points_forehead.append(triangle_coords)  # necessary for bounding box
                    if check_acceptance(index, angle_degrees, angle_history, threshold):
                        cv2.fillConvexPoly(mask_forehead_roi, triangle_coords, (
                                mapped_grayscale_value, mapped_grayscale_value, mapped_grayscale_value, cv2.LINE_AA))
                        mean_angle_forehead.append(angle_degrees)
                elif triangle in left_cheek_roi:
                    mesh_points_left_cheek.append(triangle_coords)  # necessary for bounding box
                    if check_acceptance(index, angle_degrees, angle_history, threshold):
                        cv2.fillConvexPoly(mask_left_cheek_roi, triangle_coords, (
                                mapped_grayscale_value, mapped_grayscale_value, mapped_grayscale_value, cv2.LINE_AA))
                        mean_angle_left_cheek.append(angle_degrees)
                elif triangle in right_cheek_roi:
                    mesh_points_right_cheek.append(triangle_coords)  # necessary for bounding box
                    if check_acceptance(index, angle_degrees, angle_history, threshold):
                        cv2.fillConvexPoly(mask_right_cheek_roi, triangle_coords, (
                                mapped_grayscale_value, mapped_grayscale_value, mapped_grayscale_value, cv2.LINE_AA))
                        mean_angle_right_cheek.append(angle_degrees)
            else:
                # Extract the coordinates of the three landmarks of the triangle
                triangle_coords = get_triangle_coords(image, landmark_coords_xyz, triangle)
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_outside_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))

        if use_convex_hull:
            mask_forehead_roi = apply_convex_hull(mask_forehead_roi)
            mask_left_cheek_roi = apply_convex_hull(mask_left_cheek_roi)
            mask_right_cheek_roi = apply_convex_hull(mask_right_cheek_roi)

        # calculate pixel area of optimal_roi
        if roi_mode == "optimal_roi":
            mask_optimal_roi = mask_forehead_roi + mask_left_cheek_roi + mask_right_cheek_roi
            mesh_points_bounding_box_ = mesh_points_forehead + mesh_points_left_cheek + mesh_points_right_cheek
        elif roi_mode == "forehead":
            mask_optimal_roi = mask_forehead_roi
            mesh_points_bounding_box_ = mesh_points_forehead
        elif roi_mode == "left_cheek":
            mask_optimal_roi = mask_left_cheek_roi
            mesh_points_bounding_box_ = mesh_points_left_cheek
        elif roi_mode == "right_cheek":
            mask_optimal_roi = mask_right_cheek_roi
            mesh_points_bounding_box_ = mesh_points_right_cheek
        else:
            raise Exception("No valid roi_mode selected. Valid roi_mode are: 'optimal_roi', 'forehead', 'left_cheek', 'right_cheek'.")

        if skin_segmentation_mask is not None:
            mask_optimal_roi = cv2.bitwise_and(mask_optimal_roi, mask_optimal_roi, mask=skin_segmentation_mask)

        if constrain_roi and use_outside_roi:
            interpolated_surface_normal_angles = interpolate_surface_normal_angles_scipy(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max)
            mask_eyes = mask_eyes_out(mask_outside_roi, landmark_coords_xyz)
            # extract smallest interpolation angles and create new mask only including pixels with the same amount as mask_optimal_roi
            mask_outside_roi = extract_mask_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, mask_eyes, x_min, y_min)

        # save pixel areas and mean angles of optimal ROIs in a csv file
        if constrain_roi and roi_mode == "optimal_roi":
            global csv_writer
            csv_writer.writerow([video_frame_count,
                                 count_pixel_area(mask_optimal_roi + mask_outside_roi),
                                 count_pixel_area(mask_optimal_roi),
                                 count_pixel_area(mask_outside_roi),
                                 count_pixel_area(mask_outside_roi) - count_pixel_area(mask_optimal_roi),
                                 count_pixel_area(mask_forehead_roi),
                                 count_pixel_area(mask_left_cheek_roi),
                                 count_pixel_area(mask_right_cheek_roi),
                                 round(np.average(mean_angle_forehead), 3),
                                 round(np.average(mean_angle_left_cheek), 3),
                                 round(np.average(mean_angle_right_cheek), 3)])

        return mesh_points_bounding_box_, mask_optimal_roi, mask_outside_roi

    def low_pass_filter_landmarks(self, video_frames, fps, confidence=0.5):
        mp_face_mesh = mp.solutions.face_mesh

        last_valid_coords = None  # Initialize a variable to store the last valid coordinates

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence
        ) as face_mesh:
            for rgb_frame in video_frames:
                if confidence==0.5:
                    rgb_frame = cv2.flip(rgb_frame, 1)
                else:
                    rgb_frame = rgb_frame

                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:

                        # Extract landmarks' xyz-coordinates from the detected face
                        for index, landmark in enumerate(face_landmarks.landmark):
                            x, y, z = landmark.x, landmark.y, landmark.z
                            landmark_coords_xyz_history[index].append((x, y, z))
                            last_valid_coords = (x, y, z)
                else:
                    # No face detected, append the last valid coordinates (if available)
                    if last_valid_coords is not None:
                        for index in np.arange(len(landmark_coords_xyz_history)):
                            landmark_coords_xyz_history[index].append(last_valid_coords)

        # define lowpass filter with 3.5 Hz cutoff frequency
        b, a = iirfilter(20, Wn=3.5, fs=fps, btype="low", ftype="butter")

        for idx in np.arange(478):
            try:
                if len(landmark_coords_xyz_history[idx]) > 15:  # filter needs at least 15 values to work
                    # apply filter forward and backward using filtfilt
                    x_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[0] for coords_xyz in landmark_coords_xyz_history[idx]]))
                    y_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[1] for coords_xyz in landmark_coords_xyz_history[idx]]))
                    z_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[2] for coords_xyz in landmark_coords_xyz_history[idx]]))

                    landmark_coords_xyz_history[idx] = [(x_coords_lowpass_filtered[i], y_coords_lowpass_filtered[i], z_coords_lowpass_filtered[i]) for i in
                                                        np.arange(0, len(x_coords_lowpass_filtered))]
            except ValueError:
                landmark_coords_xyz_history[idx] = landmark_coords_xyz_history[idx]

        return np.array(landmark_coords_xyz_history, dtype=np.float64)

    def roi_segmentation_for_video(self, video_frames, saved_filename, use_roi_segmentation, threshold=90, roi_mode="optimal_roi", use_convex_hull=True, constrain_roi=True, use_outside_roi=False, apply_heatmap=False):
        if use_roi_segmentation:

            if roi_mode == "optimal_roi":
                # Create a CSV file for storing pixel area data
                csv_filename = f"./data/ROI_area_log_all_datasets/{self.cached_path.split('/')[-1]}/{roi_mode}_{saved_filename}.csv"

                if not os.path.exists(f"./data/ROI_area_log_all_datasets/{self.cached_path.split('/')[-1]}"):
                    os.makedirs(f"./data/ROI_area_log_all_datasets/{self.cached_path.split('/')[-1]}")

                csv_file = open(csv_filename, mode='w', newline='')
                global csv_writer
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    ['Frame Number', 'Total Pixel Area', 'Optimal Pixel Area', 'Outside Pixel Area', 'Difference', 'Forehead', 'Left Cheek', 'Right Cheek',
                     'mean_angle_forehead', 'mean_angle_left_cheek', 'mean_angle_right_cheek'])


            # Code only used for logging the code runtime and to create a runtime plot
            if roi_mode == "optimal_roi":
                # Create a CSV file for storing pixel area data
                csv_filename_runtime = f"./data/runtime_log/{self.cached_path.split('/')[-1]}/{roi_mode}_{saved_filename}.csv"

                if not os.path.exists(f"./data/runtime_log/{self.cached_path.split('/')[-1]}"):
                    os.makedirs(f"./data/runtime_log/{self.cached_path.split('/')[-1]}")

                csv_file_runtime = open(csv_filename_runtime, mode='w', newline='')
                global csv_writer_runtime
                csv_writer_runtime = csv.writer(csv_file_runtime)
                csv_writer_runtime.writerow(
                    ['start_time', 'runtime_since_start', 'runtime_for_frame'])

            mp_face_mesh = mp.solutions.face_mesh

            fps = self.config_data.FS

            frames = list()
            max_dim_roi = 0

            global video_frame_count
            video_frame_count = 0
            global landmark_coords_xyz_history
            landmark_coords_xyz_history = [[] for _ in np.arange(478)]
            global angle_history
            angle_history = np.array([np.zeros(5) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))])

            confidence = 0.5
            if self.config_data['DATASET'] == 'KISMED':
                # these subjects need a lower detection confidence for mediapipe to work without detection errors
                if "p009" in saved_filename or "p010" in saved_filename:
                    confidence = 0.3

            landmark_coords_xyz_history = self.low_pass_filter_landmarks(video_frames, fps, confidence)

            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=confidence,
                    min_tracking_confidence=confidence
            ) as face_mesh:
                start_time = time.time()
                for rgb_frame in video_frames:
                    start_time_frame = time.time()

                    if confidence == 0.5:
                        rgb_frame = cv2.flip(rgb_frame, 1)
                    else:
                        # subjects p009 and p010 of KISMED dataset MUST NOT be NOT flipped for mediapipe to work without detection errors
                        rgb_frame = rgb_frame


                    results = face_mesh.process(rgb_frame)

                    frame = cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR)
                    mask_roi = None
                    output_roi_face = None

                    if results.multi_face_landmarks:
                        # skin segmentation
                        perform_skin_segmentation = False

                        if perform_skin_segmentation:
                            for face_landmarks in results.multi_face_landmarks:
                                # segment face region from camera frame
                                face_mask = generate_face_mask(face_landmarks, frame)
                                face_roi = cv2.bitwise_and(frame, frame, mask=face_mask)

                                # histogram based skin segmentation of facial region
                                skin_segmentation_mask = skin_segmentation(face_roi)
                                frame = cv2.bitwise_and(face_roi, face_roi, mask=skin_segmentation_mask)
                        else:
                            skin_segmentation_mask = None

                        # define mesh points and mask of each ROI if mesh triangles are below threshold
                        try:
                            if not apply_heatmap:
                                mesh_points_bounding_box_, mask_optimal_roi, mask_outside_roi = self.calculate_roi(
                                                                                        frame,
                                                                                        skin_segmentation_mask,
                                                                                        threshold=threshold,
                                                                                        roi_mode=roi_mode,
                                                                                        constrain_roi=constrain_roi,
                                                                                        use_convex_hull=use_convex_hull,
                                                                                        use_outside_roi=use_outside_roi)
                            else:
                                mesh_points_bounding_box_, mask_optimal_roi, mask_outside_roi = self.calculate_roi_heatmap(
                                                                                        frame,
                                                                                        skin_segmentation_mask,
                                                                                        threshold=threshold,
                                                                                        roi_mode=roi_mode,
                                                                                        constrain_roi=constrain_roi,
                                                                                        use_convex_hull=False,
                                                                                        use_outside_roi=use_outside_roi)
                        except IndexError as ie:
                            print(f"Index Error during processing subject {saved_filename}:  {str(video_frame_count)}")
                            print(str(ie))

                        mask_roi = mask_outside_roi if use_outside_roi else mask_optimal_roi + mask_outside_roi

                        output_roi_face = cv2.copyTo(frame, mask_roi)

                        # crop frame to square bounding box. The margins are either the outermost mesh point coordinates or (filtered) landmark coordinates
                        # use interpolated pixel ROI
                        if use_outside_roi:
                            # use filtered landmarks for a smoothed bounding box when using outside ROI during video processing
                            x_min, y_min, x_max, y_max = get_bounding_box_coordinates_filtered(output_roi_face, landmark_coords_xyz_history, video_frame_count)
                        else:
                            if constrain_roi:
                                # Use outermost coordinates of mesh points of the active ROI
                                if self.dataset_name == 'unsupervised':
                                    x_min, y_min, x_max, y_max = get_bounding_box_coordinates_mesh_points(np.array(mesh_points_bounding_box_))
                                else:
                                    # when training a neural network, the bounding box needs to be applied to the whole face
                                    # area and cannot be zoomed to the segmented ROI.
                                    # This ensures, that the ROI segmented frames overlap with the frames of the whole face
                                    x_min, y_min, x_max, y_max = get_bounding_box_coordinates_filtered(output_roi_face,
                                                                                                   landmark_coords_xyz_history,
                                                                                                   video_frame_count)
                            else:
                                # Use filtered landmarks for a smoothed bounding box of the whole face during video processing
                                x_min, y_min, x_max, y_max = get_bounding_box_coordinates_filtered(output_roi_face,
                                                                                                   landmark_coords_xyz_history,
                                                                                                   video_frame_count)

                        bb_offset = 2  # apply offset to the borders of bounding box

                        if not apply_heatmap:
                            output_roi_face, x_max_bb, x_min_bb, y_max_bb, y_min_bb = apply_bounding_box(output_roi_face,
                                                                                                         bb_offset,
                                                                                                         x_max, x_min,
                                                                                                         y_max, y_min)
                        else:
                            output_roi_face, x_max_bb, x_min_bb, y_max_bb, y_min_bb = apply_bounding_box(
                                mask_roi,
                                bb_offset,
                                x_max, x_min,
                                y_max, y_min)
                    else:
                        # use last valid data, even when no face is present in the current frame
                        if mask_roi is not None and mask_roi.any():
                            if not apply_heatmap:
                                output_roi_face = cv2.copyTo(frame, mask_roi)
                            else:
                                output_roi_face = mask_roi

                            output_roi_face = output_roi_face[int(y_min_bb):int(y_max_bb), int(x_min_bb):int(x_max_bb)]

                    if output_roi_face is not None and output_roi_face.any():
                        # append frame to list of all video frames
                        try:
                            frame = cv2.cvtColor(output_roi_face, cv2.COLOR_BGR2RGB)
                        except cv2.error as cv_e:
                            print(f"cv2 Error during processing subject {saved_filename}:  {str(video_frame_count)}")
                            print(str(cv_e))
                            pass

                    # find maximum shape of all frames
                    max_dim_frame = max(frame.shape[0], frame.shape[1])
                    if max_dim_frame > max_dim_roi:
                        max_dim_roi = max_dim_frame
                    frames.append(np.asarray(frame, dtype=np.uint8))

                    video_frame_count += 1

                    # Code only used for logging the code runtime and to create a runtime plot
                    if roi_mode == "optimal_roi":
                        end_time_frame = time.time()
                        csv_writer_runtime.writerow([start_time, str(end_time_frame - start_time), str(end_time_frame - start_time_frame)])


            if roi_mode == "optimal_roi":
                csv_file.close()
                # Code only used for logging the code runtime and to create a runtime plot
                csv_file_runtime.close()

            try:
                # resize all roi frames to same shape
                for idx in range(len(frames)):
                    frames[idx] = cv2.resize(frames[idx], (max_dim_roi, max_dim_roi))
                    frames[idx] = cv2.resize(frames[idx], (72, 72), interpolation=cv2.INTER_AREA)

                frames = np.asarray(frames, dtype=np.float64)

                return frames
            except cv2.error as cv_e2:
                print(str(cv_e2))
        else:
            return video_frames


    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """Face detection on a single frame.

        Args:
            frame(np.array): a single frame.
            backend(str): backend to utilize for face detection.
            use_larger_box(bool): whether to use a larger bounding box on face detection.
            larger_box_coef(float): Coef. of larger box.
        Returns:
            face_box_coor(List[int]): coordinates of face bouding box.
        """
        if backend == "HC":
            # Use OpenCV's Haar Cascade algorithm implementation for face detection
            # This should only utilize the CPU
            detector = cv2.CascadeClassifier(
                    './dataset/haarcascade_frontalface_default.xml')

            # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
            # (x,y) corresponds to the top-left corner of the zone to define using
            # the computed width and height.
            face_zone = detector.detectMultiScale(frame)

            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
            elif len(face_zone) >= 2:
                # Find the index of the largest face zone
                # The face zones are boxes, so the width and height are the same
                max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
                face_box_coor = face_zone[max_width_index]
                print("Warning: More than one faces are detected. Only cropping the biggest one.")
            else:
                face_box_coor = face_zone[0]
        elif backend == "RF":
            # Use a TensorFlow-based RetinaFace implementation for face detection
            # This utilizes both the CPU and GPU
            res = RetinaFace.detect_faces(frame)

            if len(res) > 0:
                # Pick the highest score
                highest_score_face = max(res.values(), key=lambda x: x['score'])
                face_zone = highest_score_face['facial_area']

                # This implementation of RetinaFace returns a face_zone in the
                # form [x_min, y_min, x_max, y_max] that corresponds to the
                # corners of a face zone
                x_min, y_min, x_max, y_max = face_zone

                # Convert to this toolbox's expected format
                # Expected format: [x_coord, y_coord, width, height]
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min

                # Find the center of the face zone
                center_x = x + width // 2
                center_y = y + height // 2

                # Determine the size of the square (use the maximum of width and height)
                square_size = max(width, height)

                # Calculate the new coordinates for a square face zone
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]
            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        else:
            raise ValueError("Unsupported face detection backend!")

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor

    def crop_face_resize(self, frames, use_face_detection, backend, use_larger_box, larger_box_coef, use_dynamic_detection,
                         detection_freq, use_median_box, width, height):
        """Crop face and resize frames.

        Args:
            frames(np.array): Video frames.
            use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                         and resizing.
                                         If True, it performs face detection every "detection_freq" frames.
            detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
            width(int): Target width for resizing.
            height(int): Target height for resizing.
            use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
            use_face_detection(bool):  Whether crop the face.
            larger_box_coef(float): the coefficient of the larger region(height and weight),
                                the middle point of the detected region will stay still during the process of enlarging.
        Returns:
            resized_frames(list[np.array(float)]): Resized and cropped frames
        """
        # Face Cropping
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq)
        else:
            num_dynamic_det = 1
        face_region_all = []
        # Perform face detection by num_dynamic_det" times.
        for idx in range(num_dynamic_det):
            if use_face_detection:
                face_region_all.append(self.face_detection(frames[detection_freq * idx], backend, use_larger_box, larger_box_coef))
            else:
                face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
        face_region_all = np.asarray(face_region_all, dtype='int')
        if use_median_box:
            # Generate a median bounding box based on all detected face regions
            face_region_median = np.median(face_region_all, axis=0).astype('int')

        # Frame Resizing
        resized_frames = np.zeros((frames.shape[0], height, width, 3))
        for i in range(0, frames.shape[0]):
            frame = frames[i]
            if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
                reference_index = i // detection_freq
            else:  # use the first region obtrained from the first frame.
                reference_index = 0
            if use_face_detection:
                if use_median_box:
                    face_region = face_region_median
                else:
                    face_region = face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                        max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames

    def chunk(self, frames, bvps, chunk_length):
        """Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of face cropped frames
            bvp_clips: all chunks of bvp frames
        """

        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    def save(self, frames_clips, bvps_clips, filename):
        """Save all the chunked data.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            count: count of preprocessed data
        """

        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count

    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=4):
        """Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(Dict): a dictionary of preprocessing configurations
            multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs)
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = Manager()  # multi-process manager
        file_list_dict = manager.dict()  # dictionary for all processes to store processed files
        p_list = []  # list of processes
        running_num = 0  # number of running processes

        # in range of number of files to process
        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < multi_process_quota:  # in case of too many processes
                    # send data to be preprocessing task
                    p = Process(target=self.preprocess_dataset_subprocess,
                                args=(data_dirs,config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict

    def build_file_list(self, file_list_dict):
        """Build a list of files used by the dataloader for the data split. Eg. list of files used for
        train / val / test. Also saves the list to a .csv file.

        Args:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """
        file_list = []
        # iterate through processes and add all processed file paths
        for process_num, file_paths in file_list_dict.items():
            file_list = file_list + file_paths

        if not file_list:
            raise ValueError(self.dataset_name, 'No files in file list')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """ If a file list has not already been generated for a specific data split build a list of files
        used by the dataloader for the data split. Eg. list of files used for
        train / val / test. Also saves the list to a .csv file.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """

        # get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.npy".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(
                1, input_signal.shape[0], target_length), np.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)
