"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import datetime
import logging
import os
import csv
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
import torch
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from tqdm import tqdm


def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_all = []

    # Create a CSV file for predicted and ground truth heart rate data for each video respectively window, if USE_SMALLER_WINDOW is chosen
    video_file = data_loader['unsupervised'].dataset.cached_path
    roi_mode = config.UNSUPERVISED.DATA.PREPROCESS.ROI_SEGMENTATION.ROI_MODE

    hr_log_path = f"./data/HR_log/{os.path.splitext(video_file)[0].split('/')[-1]}"
    hr_log_path = hr_log_path.replace(roi_mode, "optimal_roi")

    if not os.path.exists(hr_log_path):
        os.makedirs(hr_log_path)

    csv_filename = f"{hr_log_path}/{roi_mode}_{method_name}.csv"

    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["method_name", "video_file", "starting_frame", "ending_frame", "predict_hr_fft_all", "gt_hr_fft_all", "SNR_all"])

    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    gt_hr_peak_all.append(gt_hr)
                    predict_hr_peak_all.append(pre_hr)
                    SNR_all.append(SNR)
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    gt_hr_fft_all.append(gt_fft_hr)
                    predict_hr_fft_all.append(pre_fft_hr)
                    SNR_all.append(SNR)

                    csv_writer.writerow([method_name, data_loader['unsupervised'].dataset.inputs[_].split("\\")[-1], i, i+window_frame_size, pre_fft_hr, gt_fft_hr, SNR])

                    #if _ == 3:
                    #    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
                    #    BVP_window = -BVP_window
                    #    ground_truth = scaler.fit_transform(label_window.reshape(-1, 1))
                    #    BVP = scaler.fit_transform(BVP_window.reshape(-1, 1))[10:]
                    #
                    #    time = np.arange(len(BVP)) / config.UNSUPERVISED.DATA.FS
                    #
                    #    fig, ax1 = plt.subplots()
                    #    ax1.plot(time, ground_truth[10:], label='GT BVP', color='#1f77b4')
                    #    ax1.set_xlabel("Time (s)", fontsize=14)
                    #    ax1.set_ylabel("Ground Truth BVP Normalized", color='#1f77b4', fontsize=14)
                    #    ax1.tick_params('y', colors='#1f77b4', labelsize=11)
                    #    ax1.tick_params(axis='x', labelsize=11)
                    #
                    #    ax2 = ax1.twinx()
                    #    ax2.plot(time, BVP, label=method_name + ' BVP', color='#ff7f0e')
                    #    ax2.set_xlabel("Time (s)", fontsize=14)
                    #    ax2.set_ylabel(method_name + " BVP Normalized", color='#ff7f0e', fontsize=14)
                    #    ax2.tick_params('y', colors='#ff7f0e', labelsize=11)
                    #    fig.tight_layout()
                    #
                    #    plt.grid(True)
                    #    plt.show()
                else:
                    raise ValueError("Inference evaluation method name wrong!")

    # csv_writer.writerow([predict_hr_fft_all, gt_hr_fft_all, SNR_all])
    csv_file.close()

    print("Used Unsupervised Method: " + method_name)
    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "Accuracy":
                # Convert lists to NumPy arrays
                gt_hr_peak_all = np.array(gt_hr_peak_all)
                gt_hr_peak_all = np.array(gt_hr_peak_all)

                # Calculate absolute difference between the arrays
                abs_diff = np.abs(predict_hr_peak_all - gt_hr_peak_all)

                # Calculate the maximum allowed difference (currently: 1 bpm, alternatively: 1% of the corresponding value in gt_hr_peak_all)
                max_allowed_diff = 1         # 0.01 * gt_hr_peak_all

                # Check which elements have an absolute difference less than the maximum allowed difference
                correct_predictions = abs_diff <= max_allowed_diff

                # Calculate accuracy by counting the number of correct predictions and dividing by the total samples
                Accuracy = 100 * np.sum(correct_predictions) / num_test_samples
                #     # Accuracy = 100*(Summe von predict_hr_fft_all - gt_hr_fft_all < 0.02*gt_hr_fft_all) / num_test_samples
                #     Accuracy_FFT = np.mean(SNR_all)
                # standard_error = 100* np.std(Accuracy) / np.sqrt(num_test_samples)
                print("FFT Accuracy (Peak Label): {0}".format(Accuracy))

            # ToDo: regressions konfusionsmatrix erstellen: https://medium.com/@dave.cote.msc/experimenting-confusion-matrix-for-regression-a-powerfull-model-analysis-tool-7c288d99d437
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
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
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "Accuracy":
                # Convert lists to NumPy arrays
                predict_hr_fft_all = np.array(predict_hr_fft_all)
                gt_hr_fft_all = np.array(gt_hr_fft_all)

                # Calculate absolute difference between the arrays
                abs_diff = np.abs(predict_hr_fft_all - gt_hr_fft_all)

                # Calculate the maximum allowed difference (currently: 2 bpm like the accuracy of a fingerclip PPG)
                max_allowed_diff = 2

                # Check which elements have an absolute difference less than the maximum allowed difference
                correct_predictions = abs_diff <= max_allowed_diff

                # Calculate accuracy by counting the number of correct predictions and dividing by the total samples
                Accuracy = 100 * np.sum(correct_predictions) / num_test_samples
                #     # Accuracy = 100*(Summe von predict_hr_fft_all - gt_hr_fft_all < 0.02*gt_hr_fft_all) / num_test_samples
                #     Accuracy_FFT = np.mean(SNR_all)
                # standard_error = 100* np.std(Accuracy) / np.sqrt(num_test_samples)
                print("FFT Accuracy (FFT Label): {0}".format(Accuracy))

            # ToDo: regressions konfusionsmatrix erstellen: https://medium.com/@dave.cote.msc/experimenting-confusion-matrix-for-regression-a-powerfull-model-analysis-tool-7c288d99d437

            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
