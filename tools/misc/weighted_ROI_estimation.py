import argparse
import math
import os

import numpy as np
import pandas as pd
import copy
import statistics


def calc_combined_weights(angles, pixel_area_weights):
    """
       Calculate the weights for each ROI (forehead, left cheek, right cheek), based on their ROI pixel area
       and mean angle during the video.

       Parameters:
       - angles (numpy.ndarray): A 2D array where each row represents the angles associated with each ROI
       - pixel_area_weights (numpy.ndarray): A 2D array where each row represents the pixel area weights for each ROI

       Returns:
       - combined_weights: (numpy.ndarray): Numpy array with relative weights of each ROI
                                            in range (0,1) and a total sum of 1.
       """
    # Calculate column-wise mean values of angles
    mean_angles = np.mean(angles, axis=0)  # np.square(angles)  # np.power(angles, 4)

    # angle_threshold = 30
    # mean_angles[mean_angles > angle_threshold] = math.inf

    # Calculate the weights based on angles (lowest angle has highest weight)
    angle_weights = 1 / mean_angles
    angle_weights /= angle_weights.sum()

    # Calculate column-wise mean values of pixel_area_weights
    mean_pixel_area_weights = np.mean(pixel_area_weights, axis=0)
    # Normalize the pixel area weights by the total pixel area of the three ROIs (largest area has highest weight)
    mean_pixel_area_weights /= mean_pixel_area_weights.sum()

    # Calculate the combined weights by averaging the angle weights and pixel area weights elementwise
    combined_weights = (angle_weights + mean_pixel_area_weights) / 2

    return combined_weights


def calc_lowest_angle_avg_hr(pred_hr, angles, angle_threshold=30):
    """
    Calculate the weighted average heart rate from predicted HR by each ROI (forehead, left cheek, right cheek)
    based on their predicted heart rate and weights according to their ROI pixel area and mean angle during the video.

    Parameters:
    - pred_hr (numpy.ndarray): An array containing predicted heart rates for each ROI, where the columns belog to each ROI (forehead, left cheek, right cheek)
    - angles (numpy.ndarray): A 2D array where each row represents the angles associated with each ROI
    - pixel_area_weights (numpy.ndarray): A 2D array where each row represents the pixel area weights for each ROI

    Returns:
    - weighted_average: float: The weighted average heart rate of the combined ROIs, weighted by their angles and pixel areas
    """
    # Calculate column-wise mean values of angles
    mean_angles = np.mean(angles, axis=0)  # np.square(angles)  # np.power(angles, 4)

    # consider only ROIs with a mean angle below threshold of 30°
    mean_angles[mean_angles > angle_threshold] = math.inf

    # Calculate the weights based on angles (lowest angle has highest weight)
    angle_weights = 1 / mean_angles
    angle_weights /= angle_weights.sum()

    # Calculate the weighted average of the three ROI's predicted heart rate weighted by their average angles and pixel area
    weighted_average = round(np.dot(pred_hr, angle_weights), 6)

    return weighted_average, angle_weights


def calc_weighted_avg_hr(pred_hr, angles, pixel_area_weights):
    """
    Calculate the weighted average heart rate from predicted HR by each ROI (forehead, left cheek, right cheek)
    based on their predicted heart rate and weights according to their ROI pixel area and mean angle during the video.

    Parameters:
    - pred_hr (numpy.ndarray): An array containing predicted heart rates for each ROI, where the columns belog to each ROI (forehead, left cheek, right cheek)
    - angles (numpy.ndarray): A 2D array where each row represents the angles associated with each ROI
    - pixel_area_weights (numpy.ndarray): A 2D array where each row represents the pixel area weights for each ROI

    Returns:
    - weighted_average: float: The weighted average heart rate of the combined ROIs, weighted by their angles and pixel areas
    """
    # Calculate column-wise mean values of angles
    mean_angles = np.mean(angles, axis=0)  # np.square(angles)  # np.power(angles, 4)
    # Calculate the weights based on angles (lowest angle has highest weight)
    angle_weights = 1 / mean_angles
    angle_weights /= angle_weights.sum()

    # Calculate column-wise mean values of pixel_area_weights
    mean_pixel_area_weights = np.mean(pixel_area_weights, axis=0)  # np.square(pixel_area_weights)  # np.power(pixel_area_weights, 4)
    # Normalize the pixel area weights by the total pixel area of the three ROIs (largest area has highest weight)
    mean_pixel_area_weights /= mean_pixel_area_weights.sum()

    # Calculate the combined weights by averaging the angle weights and pixel area weights elementwise
    combined_weights = (angle_weights + mean_pixel_area_weights) / 2
    # Calculate the weighted average of the three ROI's predicted heart rate weighted by their average angles and pixel area
    weighted_average = round(np.dot(pred_hr, combined_weights), 6)

    return weighted_average


def predicted_hr_majority_voting(pred_hr):
    return statistics.mode(pred_hr)


def load_roi_area_log(roi_area_log_path, area_logs, subject_index, starting_frame, ending_frame):
    # iterate over all area_log_files in dataset to find the area_log_file for current subject_index
    for area_log_file in area_logs:
        # load dataframe, if the current subject string is in the area_log_filename
        if subject_index in area_log_file:
            roi_area_log = pd.read_csv(os.path.join(roi_area_log_path, area_log_file))

            # subset roi_area according to current window slice
            return roi_area_log[starting_frame:ending_frame]


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


def calculate_metrics(predict_hr_fft_all, gt_hr_fft_all, SNR_all):
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    SNR_all = np.array(SNR_all)
    num_test_samples = len(predict_hr_fft_all)

    for metric in ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'Accuracy', 'IBI']:
        if metric == "MAE":
            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
            standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
            print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
        elif metric == "RMSE":
            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
            standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
            print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
        elif metric == "MAPE":
            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(
                num_test_samples) * 100
            print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
        elif metric == "Pearson":
            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
            correlation_coefficient = Pearson_FFT[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))
            print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
        elif metric == "SNR":
            SNR_FFT = np.mean(SNR_all)
            standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
            print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
        elif metric == "Accuracy":
            # Calculate absolute difference between the arrays
            abs_diff = np.abs(predict_hr_fft_all - gt_hr_fft_all)

            # Calculate the maximum allowed difference (currently: 2 bpm like the accuracy of a fingerclip PPG)
            max_allowed_diff = 1

            # Check which elements have an absolute difference less than the maximum allowed difference
            correct_predictions = abs_diff <= max_allowed_diff

            # Calculate accuracy by counting the number of correct predictions and dividing by the total samples
            Accuracy = 100 * np.sum(correct_predictions) / num_test_samples

            print("FFT Accuracy (FFT Label): {0}".format(Accuracy))
        elif metric == "IBI":
            print("\n")
            print(round(MAE_FFT, 2))
            print(round(RMSE_FFT, 2))
            print(round(correlation_coefficient, 4))
            print(round(SNR_FFT, 2))
            print(round(Accuracy, 2))
            print("\n")


def main(args):
    hr_log_path = "./data/HR_log/" + args.folder
    roi_area_log_path = "./data/ROI_area_log/" + args.folder

    area_logs = os.listdir(roi_area_log_path)
    hr_logs = os.listdir(hr_log_path)

    methods = ["ICA", "POS", "CHROM", "GREEN", "LGI", "PBV"]
    roi_modes = ["forehead", "left_cheek", "right_cheek"]

    # iterate over all methods, subjects and video subwindows and calculate the weighted average
    # of the predicted heartrate of the three ROIs (forehead, left cheek, right cheek)
    for method in methods:
        for avg_method in ["weighted_average", "weighted_by_angle", "lowest_angle", "majority_voting"]:  # , "second_largest_hr"]:
            predict_hr_fft_all = []
            gt_hr_fft_all = []
            SNR_all = []

            combined_df = get_combined_df(hr_log_path, hr_logs, method, roi_modes)

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

                    roi_weights = calc_combined_weights(angles, pixel_area_weights)

                    SNR_list = [video_window_slice["forehead_SNR_all"], video_window_slice["left_cheek_SNR_all"],
                                video_window_slice["right_cheek_SNR_all"]]

                    if avg_method == "weighted_average":
                        # weighted average according to ROI sizes and mean angles during the video window
                        weighted_avg_hr = calc_weighted_avg_hr(pred_hr, angles, pixel_area_weights)
                        SNR_combined_dB_averaged = 10 * math.log10(
                            (roi_weights[0] * (10 ** (video_window_slice["forehead_SNR_all"] / 10)))
                            + (roi_weights[1] * (10 ** (video_window_slice["left_cheek_SNR_all"] / 10)))
                            + (roi_weights[2] * (10 ** (video_window_slice["right_cheek_SNR_all"] / 10))))
                    elif avg_method == "majority_voting":
                        # majority voting
                        # if all three HR estimations differ, consider only the forehead ROIs HR and SNR
                        weighted_avg_hr = predicted_hr_majority_voting(pred_hr)

                        # calculate SNR differently according to number of different predicted heart rates
                        if len(set(pred_hr)) == 1:
                            # mean value of SNR values
                            SNR_combined_dB_averaged = 10 * math.log10(
                                    (1/3 * (10 ** (video_window_slice["forehead_SNR_all"] / 10)))
                                    + (1/3 * (10 ** (video_window_slice["left_cheek_SNR_all"] / 10)))
                                    + (1/3 * (10 ** (video_window_slice["right_cheek_SNR_all"] / 10))))
                        elif len(set(pred_hr)) == 2:
                            # mean value of the two SNR values with the same heart rate prediction
                            indices_of_same_hr_values = np.where(pred_hr == weighted_avg_hr)[0] # np.argpartition(pred_hr, -2)[-2:]
                            SNR_combined_dB_averaged = 10 * math.log10(
                                (0.5 * (10 ** (SNR_list[indices_of_same_hr_values[0]] / 10)))
                                + (0.5 * (10 ** (SNR_list[indices_of_same_hr_values[1]] / 10))))
                        elif len(set(pred_hr)) == 3:
                            # if all three HR estimations differ, consider only the forehead ROIs HR and SNR
                            SNR_combined_dB_averaged = SNR_list[np.argmax(np.mean(pred_hr, axis=0))]

                    elif avg_method == "weighted_by_angle":
                        # calculate weighted average of each ROI below a threshold of 30 degrees
                        weighted_avg_hr, angle_weights = calc_lowest_angle_avg_hr(pred_hr, angles, angle_threshold=30)
                        if np.isnan(weighted_avg_hr):
                            # calculate weighted average of each ROI below a threshold of 45 degrees
                            weighted_avg_hr, angle_weights = calc_lowest_angle_avg_hr(pred_hr, angles, angle_threshold=45)

                        # weighted average of SNR values, weighted by their mean angle values, if they are below threshold
                        SNR_combined_dB_averaged = 10 * math.log10(
                                (angle_weights[0] * (10 ** (video_window_slice["forehead_SNR_all"] / 10)))
                                + (angle_weights[1] * (10 ** (video_window_slice["left_cheek_SNR_all"] / 10)))
                                + (angle_weights[2] * (10 ** (video_window_slice["right_cheek_SNR_all"] / 10))))
                    elif avg_method == "lowest_angle":
                        # lowest mean angle in current window slice
                        weighted_avg_hr = pred_hr[np.argmin(np.mean(angles, axis=0))]
                        # only consider SNR value of ROI with lowest angle
                        SNR_combined_dB_averaged = [video_window_slice["forehead_SNR_all"],
                                                    video_window_slice["left_cheek_SNR_all"],
                                                    video_window_slice["right_cheek_SNR_all"]][np.argmin(np.mean(angles, axis=0))]
                    elif avg_method == "second_largest_hr":
                        # second largest heart rate in current window slice
                        weighted_avg_hr = np.sort(pred_hr)[1]
                        # only consider SNR value of ROI with second largest HR
                        SNR_combined_dB_averaged = [video_window_slice["forehead_SNR_all"],
                                                    video_window_slice["left_cheek_SNR_all"],
                                                    video_window_slice["right_cheek_SNR_all"]][list(pred_hr).index(weighted_avg_hr)]


                    gt_hr_fft_all.append(gt_hr)
                    predict_hr_fft_all.append(weighted_avg_hr)
                    SNR_all.append(SNR_combined_dB_averaged)

            print("===Unsupervised Method ( " + method + " ) Predicting ===")
            print("===Averaging Method ( " + avg_method + " )  ===")
            calculate_metrics(predict_hr_fft_all, gt_hr_fft_all, SNR_all)


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=False,
                        default="KISMED_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_ROI_segmentationTrue_Angle_threshold90_ROI_mode-optimal_roi_Use_convex_hullTrue_Constrain_roiTrue_Outside_roiFalse_unsupervised",  # "PURE_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_ROI_segmentationTrue_Angle_threshold90_ROI_mode-optimal_roi_Use_convex_hullTrue_Constrain_roiTrue_Outside_roiFalse_unsupervised",
                        type=str, help="The created config folder name of the processed dataset.")
    args = parser.parse_args()

    main(args)
