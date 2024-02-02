import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import statistics
import seaborn as sns


def load_runtime_log(roi_area_log_path, area_logs, subject_index):
    # iterate over all area_log_files in dataset to find the area_log_file for current subject_index
    for area_log_file in area_logs:
        # load dataframe, if the current subject string is in the area_log_filename
        if subject_index in area_log_file:
            roi_area_log = pd.read_csv(os.path.join(roi_area_log_path, area_log_file))

            # subset roi_area according to current window slice
            return roi_area_log[1:]


def main(args):
    runtime_log_directory = "C:/Users/Philipp Witulla/PycharmProjects/rPPG-Toolbox_Thesis/data/runtime_log/"

    methods = ["Outside_roiFalse_noNumba", "Outside_roiTrue_noNumba", "Outside_roiFalse_withNumba", "Outside_roiTrue_withNumba"]

    mean_list = []
    std_dev_list = []

    # iteriere über Methoden
    for runtime_log_folder in os.listdir(runtime_log_directory):
        if not "PURE_ROI_segmentation" in runtime_log_folder:
            continue
        print(runtime_log_folder)

        runtime_log_path = os.path.join(runtime_log_directory, runtime_log_folder)

        method_dataframes_ = []  # To store the dataframes of all videos of a method

        # iteriere über Videos
        for video_log_file in os.listdir(runtime_log_path):
            # print(video_log_file)

            video_log_path = os.path.join(runtime_log_path, video_log_file)

            df_of_method = pd.read_csv(video_log_path, header=0)
            method_dataframes_.append(df_of_method[1:])

        method_dataframe = copy.deepcopy(method_dataframes_)
        method_dataframe = pd.concat(method_dataframe, axis=0)

        frame_times_ms = method_dataframe["runtime_for_frame"] * 1000  # convert seconds to milliseconds

        method_mean = frame_times_ms.mean()
        method_std = frame_times_ms.std()
        mean_list.append(method_mean)
        std_dev_list.append(method_std)

    labels = ['no interpolation\nwith numba',
              'with interpolation\nwith numba',
              'no interpolation\nwithout numba',
              'with interpolation\nwithout numba']

    # Create an array of indices for the data points
    indices = [1, 1, 1.8, 1.8]

    # Set the font family to Charter
    font = {'family': 'serif', 'serif': ['Charter'], 'size': 12}

    # Set the font properties for the entire plot
    plt.rc('font', **font)

    # Create the grouped horizontal bar plot
    group_width = 0.3
    alpha = 1
    plt.barh(indices[0]+group_width/2, mean_list[0], xerr=std_dev_list[0], color='tab:blue', alpha=alpha, height=group_width,
             label='no interpolation\nwith numba', align='center')
    plt.barh(indices[1]-group_width/2, mean_list[1], xerr=std_dev_list[1], color='tab:orange', alpha=alpha, height=group_width,
             label='with interpolation\nwith numba', align='center')

    plt.barh(indices[2]+group_width/2, mean_list[2], xerr=std_dev_list[2], color='tab:blue', alpha=alpha, height=group_width,
             label='no interpolation\nwithout numba', align='center')
    plt.barh(indices[3]-group_width/2, mean_list[3], xerr=std_dev_list[3], color='tab:orange', alpha=alpha, height=group_width,
             label='with interpolation\nwithout numba', align='center')

    # Customize the plot
    plt.yticks([indices[0]+group_width/2, indices[1]-group_width/2, indices[2]+group_width/2, indices[3]-group_width/2], labels)
    plt.xlabel('Runtime per frame (ms)')
    plt.title('Mean and standard deviation of runtime per frame')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    # plt.legend()

    # Show the plot
    plt.show()

        # for roi_mode in roi_modes:
        #     # Find the file matching the method and roi
        #     matching_file = [hr_log_file for hr_log_file in hr_logs if
        #                      method in hr_log_file and roi_mode in hr_log_file]
        #     # load the hr_log csv-file into a dataframe
        #     hr_log_file_path = os.path.join(hr_log_path, matching_file[0])
        #     df_of_method = pd.read_csv(hr_log_file_path, header=0)
        #     optimal_roi_dataframes_.append(df_of_method)

        # optimal_roi_dataframes = copy.deepcopy(optimal_roi_dataframes_)
        #
        # for i, df in enumerate(optimal_roi_dataframes):
        #     header = roi_modes[i]
        #     # Rename the columns in the DataFrame
        #     df.columns = [f"{header}_{col}" for col in df.columns]
        # # Concatenate the DataFrames into a single one
        # combined_df = pd.concat(optimal_roi_dataframes, axis=1)
        # return combined_df



    '''
    counter = 1

    # iterate over all methods, subjects and video subwindows and calculate the weighted average
    # of the predicted heartrate of the three ROIs (forehead, left cheek, right cheek)
    for method in methods:

        combined_df = get_combined_df(runtime_log_path, runtime_logs, method, roi_modes)

        subjects_filename = combined_df["forehead_video_file"].unique()
        subjects_index = [subject.split("_")[0] for subject in combined_df["forehead_video_file"].unique()]

        # iterate over all subjects for given method
        for subject_index in subjects_index:
            # create dataframe of a videos predicted HR and frame indices of each window slice
            combined_df_filtered_by_method_and_subject = combined_df[combined_df["forehead_video_file"].str.contains(subject_index)]

            # iterate over each each window_slice, if EVALUATION_WINDOW was set during preprocessing (= iterate over each row in the dataframe )
            for index, video_window_slice in combined_df_filtered_by_method_and_subject.iterrows():
                starting_frame = video_window_slice["forehead_starting_frame"]
                ending_frame = video_window_slice["forehead_ending_frame"]

                pred_hr = combined_df_filtered_by_method_and_subject.loc[
                    [index], ["forehead_predict_hr_fft_all", "left_cheek_predict_hr_fft_all", "right_cheek_predict_hr_fft_all"]].to_numpy()[0]

                gt_hr = video_window_slice["forehead_gt_hr_fft_all"]


                # load area_log_file for current subject
                roi_area_log = load_roi_area_log(roi_area_log_path, area_logs, subject_index, starting_frame, ending_frame)

                # if a video's last window slice consists of just a few frames (defined as less than a third of a window slice) and is not suited for valid HR prediction, skip it
                if len(roi_area_log) < (ending_frame - starting_frame)/3 or roi_area_log.empty:
                    continue

                # Mean values of surface normal angles of the three rois (forehead, left_cheek, right_cheek) during current window slice
                angles = roi_area_log[['mean_angle_forehead', 'mean_angle_left_cheek', 'mean_angle_right_cheek']].to_numpy()

                # Pixel area weights of current window slice
                pixel_area_weights = roi_area_log[['Forehead', 'Left Cheek', 'Right Cheek']].to_numpy()

                counter += 1

        # df_plot = pd.melt(pd.DataFrame({"A": gt_hr_fft_all, "B": predict_hr_fft_all}), var_name='Ground Truth HR', value_name='Predicted HR')
        scatter_plot_gt_pred_hr(gt_hr_fft_all, predict_hr_fft_all)
        
    '''


def scatter_plot_gt_pred_hr(gt_hr_fft_all, predict_hr_fft_all):
    figManager = plt.get_current_fig_manager()
    figManager.window.state('zoomed')

    # scatterplot of gt_hr and pred_hr
    # list to store point colors
    colors = []
    # Count the number of matches
    num_matches = sum(1 for gt, pred in zip(gt_hr_fft_all, predict_hr_fft_all) if gt == pred)


    # plt.scatter(gt_hr_fft_all, predict_hr_fft_all, c=colors, s=point_sizes)
    plt.scatter(gt_hr_fft_all, predict_hr_fft_all, c=colors, marker='o', alpha=0.8)  # s=base_point_size
    plt.xlabel('Ground Truth Heart Rate')
    plt.ylabel('Predicted Heart Rate')
    plt.xticks(np.unique(gt_hr_fft_all))
    plt.yticks(np.unique(predict_hr_fft_all))
    plt.xticks(rotation=45)

    # Add the identity line
    plt.plot([min(gt_hr_fft_all), max(gt_hr_fft_all)], [min(gt_hr_fft_all), max(gt_hr_fft_all)],
             color='black', linestyle='--', label='Identity Line')

    # Plot the identity line with y-value offset by +2 and -2
    offset = 2
    plt.plot([min(gt_hr_fft_all), max(gt_hr_fft_all)], [min(gt_hr_fft_all) + offset, max(gt_hr_fft_all) + offset],
             color='red', linestyle='--', label='Identity Line +2 BPM')

    offset = -2
    plt.plot([min(gt_hr_fft_all), max(gt_hr_fft_all)], [min(gt_hr_fft_all) + offset, max(gt_hr_fft_all) + offset],
             color='red', linestyle='--', label='Identity Line -2 BPM')

    # Customize the legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Exact prediction',
                                  markerfacecolor='green', markersize=10),
                      # plt.Line2D([0], [0], marker='o', color='w', label='difference < 1 BPM',
                      #            markerfacecolor='yellow', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='difference < 2 BPM',
                                  markerfacecolor='orange', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='difference > 2 BPM',
                                  markerfacecolor='red', markersize=10)]
    plt.legend(handles=legend_elements)
    plt.grid()

    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.99, wspace=0.2, hspace=0.2)

    # Show the plot
    plt.show()


def get_combined_df(hr_log_path, hr_logs, method, roi_modes):
    optimal_roi_dataframes_ = []  # To store the dataframes for each ROI

    for roi_mode in roi_modes:
        # Find the file matching the method and roi
        matching_file = [hr_log_file for hr_log_file in hr_logs if method in hr_log_file and roi_mode in hr_log_file]
        # load the hr_log csv-file into a dataframe
        hr_log_file_path = os.path.join(hr_log_path, matching_file[0])
        df_of_method = pd.read_csv(hr_log_file_path, header=0)
        optimal_roi_dataframes_.append(df_of_method)

    optimal_roi_dataframes = copy.deepcopy(optimal_roi_dataframes_)

    for i, df in enumerate(optimal_roi_dataframes):
        header = roi_modes[i]
        # Rename the columns in the DataFrame
        df.columns = [f"{header}_{col}" for col in df.columns]
    # Concatenate the DataFrames into a single one
    combined_df = pd.concat(optimal_roi_dataframes, axis=1)
    return combined_df


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=False,
                        default="PURE_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_ROI_segmentationTrue_Angle_threshold90_ROI_mode-optimal_roi_Use_convex_hullTrue_Constrain_roiTrue_Outside_roiFalse_unsupervised",  # "PURE_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_ROI_segmentationTrue_Angle_threshold90_ROI_mode-optimal_roi_Use_convex_hullTrue_Constrain_roiTrue_Outside_roiFalse_unsupervised",
                        type=str, help="The created config folder name of the processed dataset.")
    args = parser.parse_args()

    main(args)
