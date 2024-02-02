"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import os
import csv
import numpy as np
import evaluation.post_process
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman


def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("\n===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_peak_all = []
    SNR_fft_all = []

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
                    # print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9 or smaller "
                    #       f"than a third of window_frame_size. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    gt_hr_peak_all.append(gt_hr)
                    predict_hr_peak_all.append(pre_hr)
                    SNR_peak_all.append(SNR)
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    gt_hr_fft_all.append(gt_fft_hr)
                    predict_hr_fft_all.append(pre_fft_hr)
                    SNR_fft_all.append(SNR)

                    csv_writer.writerow([method_name, data_loader['unsupervised'].dataset.inputs[_].split("\\")[-1], i, i+window_frame_size, pre_fft_hr, gt_fft_hr, SNR])

                    # save numpy arrays if you want to plot the signals, using the script: tools/output_signal_viz/plot_gt_and_rppg_bvp.py
                    # time = np.arange(len(BVP)) / config.UNSUPERVISED.DATA.FS
                    # np.save(f'{video_file}/time_{_}_{idx}_{i}.npy', time)
                    # np.save(f'{video_file}/ground_truth_{_}_{idx}_{i}.npy', ground_truth)
                    # np.save(f'{video_file}/{method_name}_BVP_{_}_{idx}_{i}.npy', BVP)
                else:
                    raise ValueError("Inference evaluation method name wrong!")

                # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
                if config.TOOLBOX_MODE == 'unsupervised_method':
                    filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
                else:
                    raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')

    csv_file.close()

    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_peak_all = np.array(SNR_peak_all)
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
                SNR_peak = np.mean(SNR_peak_all)
                standard_error = np.std(SNR_peak_all) / np.sqrt(num_test_samples)
                print("PEAK SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_peak, standard_error))
            elif metric == "Accuracy":
                gt_hr_peak_all = np.array(gt_hr_peak_all)
                gt_hr_peak_all = np.array(gt_hr_peak_all)

                # calculate absolute difference between the arrays
                abs_diff = np.abs(predict_hr_peak_all - gt_hr_peak_all)

                # calculate the maximum allowed difference (currently: 2 bpm, alternatively: 1% of the corresponding value in gt_hr_peak_all)
                max_allowed_diff = 2      # 0.01 * gt_hr_peak_all

                # check how much elements have an absolute difference less than the maximum allowed difference
                correct_predictions = abs_diff <= max_allowed_diff
                Accuracy = 100 * np.sum(correct_predictions) / num_test_samples
                print("PEAK Accuracy (Peak Label): {0}".format(Accuracy))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        SNR_fft_all = np.array(SNR_fft_all)
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
                SNR_fft = np.mean(SNR_fft_all)
                standard_error = np.std(SNR_fft_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_fft, standard_error))
            elif metric == "Accuracy":
                predict_hr_fft_all = np.array(predict_hr_fft_all)
                gt_hr_fft_all = np.array(gt_hr_fft_all)

                # calculate absolute difference between the arrays
                abs_diff = np.abs(predict_hr_fft_all - gt_hr_fft_all)

                # calculate the maximum allowed difference (currently: 2 bpm like the accuracy of a fingerclip PPG)
                max_allowed_diff = 2

                # Check how much elements have an absolute difference less than the maximum allowed difference
                correct_predictions = abs_diff <= max_allowed_diff
                Accuracy = 100 * np.sum(correct_predictions) / num_test_samples
                print("FFT Accuracy (FFT Label): {0}".format(Accuracy))

                print("\n")
                print(round(MAE_FFT, 2))
                print(round(RMSE_FFT, 2))
                print(round(correlation_coefficient, 4))
                print(round(SNR_fft, 2))
                print(round(Accuracy, 2))
                print("\n")
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                        x_label='Referenzherzrate (BPM)',
                        y_label='Vorhergesagte Herzrate (BPM)',
                        show_legend=True, figure_size=(5, 5),
                        the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                        file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                        x_label='Differenz zwischen vorhergesagter und Referenzherzrate (BPM)',
                        y_label='Mittelwert aus vorhergesagter und Referenzherzrate (BPM)',
                        show_legend=True, figure_size=(5, 5),
                        the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                        file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
