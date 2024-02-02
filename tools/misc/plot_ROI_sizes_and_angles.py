import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy


def load_roi_area_log(roi_area_log_path, area_logs):
    roi_area_df = None

    # iterate over all area_log_files in dataset to find the area_log_file for current subject_index
    for idx, area_log_file in enumerate(area_logs):
        # load dataframe, if the current subject string is in the area_log_filename
        # if subject_index in area_log_file:
        roi_area_log = pd.read_csv(os.path.join(roi_area_log_path, area_log_file))
        if idx == 0:
            roi_area_df = roi_area_log
        else:
            roi_area_df = pd.concat([roi_area_df, roi_area_log])


    return roi_area_df


def get_combined_df(hr_log_path, hr_logs, method, roi_modes):
    optimal_roi_dataframes_ = []  # To store the dataframes for each ROI

    for roi_mode in roi_modes:
        # Find the file matching the method and roi
        matching_file = [hr_log_file for hr_log_file in hr_logs if method in hr_log_file and roi_mode in hr_log_file]
        # load the hr_log csv-file into a dataframe
        hr_log_file_path = os.path.join(hr_log_path, matching_file[0])
        df_of_method = pd.read_csv(hr_log_file_path, header=0)
        # <df_of_method = df_of_method[df_of_method['starting_frame'] != 1200].reset_index(drop=True)
        optimal_roi_dataframes_.append(df_of_method)

    optimal_roi_dataframes = copy.deepcopy(optimal_roi_dataframes_)

    for i, df in enumerate(optimal_roi_dataframes):
        header = roi_modes[i]
        # Rename the columns in the DataFrame
        df.columns = [f"{header}_{col}" for col in df.columns]
    # Concatenate the DataFrames into a single one
    combined_df = pd.concat(optimal_roi_dataframes, axis=1)
    return combined_df


def main(args):
    mean_angles_list = []
    std_angles_list = []
    mean_pixel_areas_list = []
    std_pixel_areas_list = []

    datasets = ["UBFC-rPPG", "COHFACE", "VIPL-HR-V1", "PURE", "MMPD", "RLAP", "KISMED"]

    # iterate over all datasets
    for ds in datasets:
        print(ds)
        # + "filtered_30fps/"
        if ds == "UBFC-rPPG":
            roi_area_log_path = "./data/ROI_area_log/" + ds + args.folder
        elif ds == "MMPD":
            roi_area_log_path = "./data/ROI_area_log_all_datasets/filtered_30fps/" + ds + args.folder
        else:
            roi_area_log_path = "./data/ROI_area_log_all_datasets/" + ds + args.folder  # _all_datasets  #  + "filtered_30fps/"

        area_logs = os.listdir(roi_area_log_path)

        # load area_log_file for current subject
        roi_area_log = load_roi_area_log(roi_area_log_path, area_logs)

        # if a video's last window slice consists of just a few frames (defined as less than a third of a window slice) and is not suited for valid HR prediction, skip it
        if roi_area_log.empty:
            continue

        # Mean values of surface normal angles of the three rois (forehead, left_cheek, right_cheek) during current window slice
        angles = roi_area_log[['mean_angle_forehead', 'mean_angle_left_cheek', 'mean_angle_right_cheek']].to_numpy()
        mean_angles = np.mean(angles, axis=0)
        std_angles = np.std(angles, axis=0)

        # Pixel area weights of current window slice
        pixel_areas = roi_area_log[['Forehead', 'Left Cheek', 'Right Cheek']].to_numpy()
        if ds == "VIPL-HR-V1":
            pixel_areas = pixel_areas / 4
        # if ds == "KISMED":
        #     pixel_areas = pixel_areas * 4
        mean_pixel_areas = np.mean(pixel_areas, axis=0)
        std_pixel_areas = np.std(pixel_areas, axis=0)

        mean_angles_list.append(mean_angles)
        std_angles_list.append(std_angles)
        mean_pixel_areas_list.append(mean_pixel_areas)
        std_pixel_areas_list.append(std_pixel_areas)

    print("\nFinished processing datasets. Plotting...")
    mean_angles_arr = np.array(mean_angles_list).T
    std_angles_arr = np.array(std_angles_list).T
    mean_pixel_areas_arr = np.array(mean_pixel_areas_list).T
    std_pixel_areas_arr = np.array(std_pixel_areas_list).T


    # Set the font family to Charter
    font = {'family': 'serif', 'serif': ['Charter'], 'size': 12}

    # Set the font properties for the entire plot
    plt.rc('font', **font)

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(8, 6))

    # Set position of bar on X axis
    br1 = np.arange(len(mean_angles_arr[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, mean_angles_arr[0], yerr=std_angles_arr[0], color='tab:blue', width=barWidth,
            edgecolor='grey', label='Stirn')
    plt.bar(br2, mean_angles_arr[1], yerr=std_angles_arr[1], color='tab:orange', width=barWidth,
            edgecolor='grey', label='Linke Wange')
    plt.bar(br3, mean_angles_arr[2], yerr=std_angles_arr[2], color='tab:green', width=barWidth,
            edgecolor='grey', label='Rechte Wange')

    # Adding Xticks
    plt.xlabel('Datensatz', fontweight='bold', fontsize=13)  # , fontweight='bold'
    plt.ylabel('Mittlerer Reflexionswinkel (in °)', fontweight='bold', fontsize=13)
    plt.xticks([r + barWidth for r in range(len(mean_angles_arr[0]))],
               ["UBFC-rPPG", "COHFACE", "VIPL-HR-V1\n(Bewegungs-\nszenario)", "PURE\n(Bewegungs-\nszenario)", "MMPD\n(Kniebeugen-\nszenario)", "RLAP-rPPG\n(Videospiel-\nszenario)", "KISMED\n(Rotations-\nszenario)"])

    plt.grid(axis='y')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, top=0.99, bottom=0.19, left=0.07, right=0.99)
    plt.show()

    fig = plt.subplots(figsize=(8, 6))

    # Set position of bar on X axis
    br1 = np.arange(len(mean_pixel_areas_arr[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, mean_pixel_areas_arr[0], yerr=std_pixel_areas_arr[0], color='tab:blue', width=barWidth,
            edgecolor='grey', label='Stirn')
    plt.bar(br2, mean_pixel_areas_arr[1], yerr=std_pixel_areas_arr[1], color='tab:orange', width=barWidth,
            edgecolor='grey', label='Linke Wange')
    plt.bar(br3, mean_pixel_areas_arr[2], yerr=std_pixel_areas_arr[2], color='tab:green', width=barWidth,
            edgecolor='grey', label='Rechte Wange')

    # Adding Xticks
    plt.xlabel('Datensatz', fontweight='bold', fontsize=13)  # , fontweight='bold'
    plt.ylabel('Mittlere Anzahl an Pixel innerhalb der ROI', fontweight='bold', fontsize=13)
    plt.xticks([r + barWidth for r in range(len(mean_pixel_areas_arr[0]))],
               ["UBFC-rPPG", "COHFACE", "VIPL-HR-V1\n(Bewegungs-\nszenario)", "PURE\n(Bewegungs-\nszenario)", "MMPD\n(Kniebeugen-\nszenario)", "RLAP-rPPG\n(Videospiel-\nszenario)", "KISMED\n(Rotations-\nszenario)"])

    plt.grid(axis='y')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, top=0.99, bottom=0.14, left=0.07, right=0.99)
    plt.show()


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=False,
                        default="_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_ROI_segmentationTrue_Angle_threshold90_ROI_mode-optimal_roi_Use_convex_hullTrue_Constrain_roiTrue_Outside_roiFalse_unsupervised",  # "PURE_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_ROI_segmentationTrue_Angle_threshold90_ROI_mode-optimal_roi_Use_convex_hullTrue_Constrain_roiTrue_Outside_roiFalse_unsupervised",
                        type=str, help="The created config folder name of the processed dataset.")
    args = parser.parse_args()

    main(args)
