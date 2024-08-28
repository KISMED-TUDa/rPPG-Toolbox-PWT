"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
            (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0:  # catches divide by 0 runtime warning
        SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.5, 4.16] Hz # Note: since signal was filtered before with much lower bounds, this does not change anything
        # equals [30, 250] beats per min
        LPF = 30 / 60  # = 30 BPM
        HPF = 250 / 60  # = 250 BPM
        [b, a] = butter(1, [LPF / fs * 2, HPF / fs * 2], btype='bandpass')

        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    if hr_method == 'FFT':
        LPF = 30 / 60  # = 30 BPM
        HPF = 250 / 60  # = 250 BPM
        hr_pred = _calculate_fft_hr(predictions, fs=fs, low_pass=LPF, high_pass=HPF)
        hr_label = _calculate_fft_hr(labels, fs=fs, low_pass=LPF, high_pass=HPF)

        # plot_fft(labels, predictions, sample_rate=fs, title='FFT of ground truth and predicted BVP signal')

    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, SNR


"""
Functions to calculate SDNN and RMSSD. Currently unused in project.
"""
def calculate_sdnn_and_rmssd(pred_signal, gt_signal, fs):
    """
    Calculate SDNN (Standard Deviation of NN intervals) and RMSSD (Root Mean Square of Successive Differences)
    values of predicted and ground truth BVP signal

    Args:
    - pred_signal: List or array containing predicted BVP signal
    - gt_signal: List or array containing ground truth BVP signal
    - fs: Sample rate of the video

    Returns:
    - SDNN and RMSSD values of predicted and ground truth BVP signal
    """
    intervals_pred_seconds, intervals_gt_seconds = _calculate_IBI_in_seconds(pred_signal, gt_signal, fs)

    intervals_pred = intervals_pred_seconds * 1000
    intervals_gt = intervals_gt_seconds * 1000

    if len(intervals_pred) > 2:
        sdnn_pred = np.std(intervals_pred, ddof=1)
        sdnn_gt = np.std(intervals_gt, ddof=1)
        rmssd_pred = np.sqrt(np.mean(np.square(np.diff(intervals_pred))))
        rmssd_gt = np.sqrt(np.mean(np.square(np.diff(intervals_gt))))
    else:
        sdnn_pred = np.nan
        sdnn_gt = np.nan
        rmssd_pred = np.nan
        rmssd_gt = np.nan

    # if sdnn_pred > 100:
    #     plot_bvp_with_peaks(gt_signal, pred_signal, fs=fs, title='Tatsächliche und vorhergesagte BVP Signale')

    #    if np.isnan(sdnn_pred) or np.isnan(sdnn_gt) or np.isnan(rmssd_pred) or np.isnan(rmssd_gt):
    #        plot_bvp_with_peaks(gt_signal, pred_signal, fs=fs, title='Tatsächliche und vorhergesagte BVP Signale')

    # print(f"{sdnn_pred}, {sdnn_gt}, {rmssd_pred}, {rmssd_gt}")

    return sdnn_pred, sdnn_gt, rmssd_pred, rmssd_gt


def _calculate_peak_HRs_of_equal_length(pred_signal, gt_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    intervals_pred, intervals_gt = _calculate_IBI_in_seconds(pred_signal, gt_signal, fs)

    hr_peak = 60 / np.mean(intervals_pred)
    hr_peak_gt = 60 / np.mean(intervals_gt)
    return hr_peak, hr_peak_gt


def _calculate_IBI_in_seconds(pred_signal, gt_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    # set the prominance (= height threshold) to one fourth of the difference between max and min value of signal
    prominence_pred = 0.25 * (np.max(pred_signal) - np.min(pred_signal))
    prominence_gt = 0.25 * (np.max(gt_signal) - np.min(gt_signal))

    ppg_peaks_pred, _ = scipy.signal.find_peaks(pred_signal, prominence=prominence_pred)
    ppg_peaks_gt, _ = scipy.signal.find_peaks(gt_signal, prominence=prominence_gt)

    # reduce detection prominence limit, if there are less than three peaks or two IBI detected
    if len(ppg_peaks_pred) < 3:
        ppg_peaks_pred, _ = scipy.signal.find_peaks(pred_signal, prominence=0.03 * (np.max(pred_signal) - np.min(pred_signal)))
        if len(ppg_peaks_pred) < 3:
            ppg_peaks_pred, _ = scipy.signal.find_peaks(pred_signal)

    # find lower amount of peaks in both signals and take only this amount of ppg_peaks into consideration
    min_peak_count = min(len(ppg_peaks_pred), len(ppg_peaks_gt))

    ppg_peaks_pred = ppg_peaks_pred[:min_peak_count]
    ppg_peaks_gt = ppg_peaks_gt[:min_peak_count]

    # Calculate peak-to-peak intervals with the unit: number of frames
    intervals_pred = np.diff(ppg_peaks_pred)
    intervals_gt = np.diff(ppg_peaks_gt)

    # transform peak-to-peak intervals to the unit: seconds
    intervals_pred = intervals_pred / fs
    intervals_gt = intervals_gt / fs

    return intervals_pred, intervals_gt


def calculate_ibi_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True):
    """Calculate video-level HR and inter beat intervals"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz
        # equals [45, 150] beats per min
        # [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')

        # bandpass filter between [0.5, 4.16] Hz
        # equals [30, 250] beats per min
        LPF = 30 / 60  # = 30 BPM
        HPF = 250 / 60  # = 250 BPM
        [b, a] = butter(1, [LPF / fs * 2, HPF / fs * 2], btype='bandpass')

        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))

    mean_interval_pred, std_deviation_pred, hr_pred, hr_pred_std_dev, mean_interval_gt, std_deviation_gt, hr_gt, hr_gt_std_dev \
        = calculate_inter_beat_intervals(predictions, labels, fs)

    return mean_interval_pred, std_deviation_pred, hr_pred, hr_pred_std_dev, mean_interval_gt, std_deviation_gt, hr_gt, hr_gt_std_dev


def calculate_inter_beat_intervals(data_pred, data_gt, fs):
    intervals_pred, intervals_gt = _calculate_IBI_in_seconds(data_pred, data_gt, fs)

    # Calculate mean and standard deviation of the intervals
    mean_interval_pred = np.mean(intervals_pred)
    std_deviation_pred = np.std(intervals_pred, ddof=1)  # SDNN value
    rmssd_pred = np.sqrt(np.mean(np.square(np.diff(intervals_pred))))

    mean_interval_gt = np.mean(intervals_gt)
    std_deviation_gt = np.std(intervals_gt, ddof=1)  # SDNN value
    rmssd_gt = np.sqrt(np.mean(np.square(np.diff(intervals_gt))))

    hr_pred = 60 / mean_interval_pred
    hr_pred_std_dev = (60 / (mean_interval_pred ** 2)) * std_deviation_pred
    hr_gt = 60 / mean_interval_gt
    hr_gt_std_dev = (60 / (mean_interval_gt ** 2)) * std_deviation_gt

    return mean_interval_pred, std_deviation_pred, hr_pred, hr_pred_std_dev, mean_interval_gt, std_deviation_gt, hr_gt, hr_gt_std_dev


def plot_bvp_with_peaks(data_gt, data_pred, fs=30, title='Tatsächliche und vorhergesagte BVP Signale'):
    # set the font to Charter
    font = {'family': 'serif', 'serif': ['Charter'], 'size': 12}
    plt.rc('font', **font)

    # set the prominance (= height threshold) to one fourth of the difference between max and min value of signal
    prominence_pred = 0.25 * (np.max(data_pred) - np.min(data_pred))
    prominence_gt = 0.25 * (np.max(data_gt) - np.min(data_gt))

    ppg_peaks_pred, _ = scipy.signal.find_peaks(data_pred, prominence=prominence_pred)
    ppg_peaks_gt, _ = scipy.signal.find_peaks(data_gt, prominence=prominence_gt)

    # reduce detection prominence limit, if there are less than three peaks or two IBI detected
    if len(ppg_peaks_pred) < 3:
        ppg_peaks_pred, _ = scipy.signal.find_peaks(data_pred, prominence=0.03 * (np.max(data_pred) - np.min(data_pred)))
        if len(ppg_peaks_pred) < 3:
            ppg_peaks_pred, _ = scipy.signal.find_peaks(data_pred)

    # find lower amount of peaks in both signals and take only this amount of ppg_peaks into consideration
    min_peak_count = min(len(ppg_peaks_pred), len(ppg_peaks_gt))

    ppg_peaks_pred = ppg_peaks_pred[:min_peak_count]
    ppg_peaks_gt = ppg_peaks_gt[:min_peak_count]

    plt.figure(figsize=(8, 6))
    time = np.arange(len(data_gt)) / fs

    plt.plot(time, data_gt, 'tab:blue', label='Tatsächliches BVP Signal')
    plt.plot(time[ppg_peaks_gt], data_gt[ppg_peaks_gt], "kx")

    plt.plot(time, data_pred, 'tab:orange', label=('Vorhergesagtes BVP Signal'))
    plt.plot(time[ppg_peaks_pred], data_pred[ppg_peaks_pred], "rx")
    # Add a vertical line at the peak frequency
    # plt.axvline(x=peak_frequency, color='red', linestyle='--', label=f'Peak Frequency: {peak_frequency:.2f} Hz ({60*peak_frequency:.2f} BPM)')
    plt.title(title)
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fft(data_gt, data_pred, sample_rate=30, title='FFT of ground truth and predicted BVP signal'):
    fft_result_gt = np.fft.fft(data_gt)
    freqs_gt = np.fft.fftfreq(len(fft_result_gt), 1 / sample_rate)

    fft_result_pred = np.fft.fft(data_pred)
    freqs_pred = np.fft.fftfreq(len(fft_result_pred), 1 / sample_rate)

    # find the index of positive frequencies
    positive_indices_gt = np.where(freqs_gt > 0)
    positive_frequencies_gt = freqs_gt[positive_indices_gt]
    positive_magnitude_gt = fft_result_gt[positive_indices_gt]

    # find the index of positive frequencies
    positive_indices_pred = np.where(freqs_pred > 0)
    positive_frequencies_pred = freqs_pred[positive_indices_pred]
    positive_magnitude_pred = fft_result_pred[positive_indices_pred]

    # set the font to Charter
    font = {'family': 'serif', 'serif': ['Charter'], 'size': 12}
    plt.rc('font', **font)

    # Find the index of the peak magnitude
    magnitude = np.abs(positive_magnitude_pred)
    peak_index = np.argmax(magnitude)
    peak_frequency = freqs_pred[peak_index]

    plt.figure(figsize=(8, 6))
    plt.plot(positive_frequencies_gt, np.abs(positive_magnitude_gt), 'tab:blue', label='FFT of ground truth BVP signal')
    plt.plot(positive_frequencies_pred, np.abs(positive_magnitude_pred), 'tab:orange',
             label=('FFT of predicted BVP signal'))
    # add a vertical line at the peak frequency
    plt.axvline(x=peak_frequency, color='red', linestyle='--',
                label=f'Peak Frequency: {peak_frequency:.2f} Hz ({60 * peak_frequency:.2f} BPM)')
    # plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
