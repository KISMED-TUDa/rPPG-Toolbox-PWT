import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
import math
from scipy.stats import gaussian_kde

class BlandAltman():

    def __init__(self,gold_std,new_measure,config,averaged=False):
        # set averaged to True if multiple observations from each participant are averaged together to get one value
        import pandas as pd

        # Check that inputs are list or pandas series, convert to series if list
        if isinstance(gold_std,list) or isinstance(gold_std, (np.ndarray, np.generic) ):
            df = pd.DataFrame() # convert to pandas series
            df['gold_std'] = gold_std
            gold_std = df.gold_std
        elif not isinstance(gold_std,pd.Series):
            print('Error: Data type of gold_std is not a list or a Pandas series or Numpy array')

        if isinstance(new_measure,list) or isinstance(new_measure, (np.ndarray, np.generic) ):
            df2 = pd.DataFrame() # convert to pandas series
            df2['new_measure'] = new_measure
            new_measure = df2.new_measure
        elif not isinstance(new_measure,pd.Series):
            print('Error: Data type of new_measure is not a list or a Pandas series or Numpy array')

        self.gold_std = gold_std
        self.new_measure = new_measure

        # Calculate Bland-Altman statistics
        diffs = gold_std - new_measure
        self.mean_error = diffs.mean()
        self.std_error = diffs.std()
        self.mean_absolute_error = diffs.abs().mean()
        self.mean_squared_error = (diffs ** 2).mean()
        self.root_mean_squared_error = np.sqrt((diffs**2).mean())
        r = np.corrcoef(self.gold_std,self.new_measure)
        self.correlation = r[0,1]  # correlation coefficient
        diffs_std = diffs.std()    # 95% Confidence Intervals
        corr_std = np.sqrt(2*(diffs_std**2)) # if observations are averaged, used corrected standard deviation
        sqrt_sample_size = math.sqrt(self.gold_std.shape[0])
        if averaged:
            self.CI95 = [self.mean_error + 1.96 * corr_std , self.mean_error - 1.96 * corr_std]
        else:
            self.CI95 = [self.mean_error + 1.96 * diffs_std, self.mean_error - 1.96 * diffs_std]

        # Define save path
        if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
            self.save_path  = os.path.join(config.LOG.PATH, config.TEST.DATA.EXP_DATA_NAME, 'bland_altman_plots')
        elif config.TOOLBOX_MODE == 'unsupervised_method':
            self.save_path  = os.path.join(config.LOG.PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME, 'bland_altman_plots')
        else:
            raise ValueError('TOOLBOX_MODE only supports train_and_test, only_test, or unsupervised_method!')
        
        # Make the save path, if needed
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def print_stats(self,round_amount = 5):
        print("Mean error = {}".format(round(self.mean_error,round_amount)))
        print("Mean absolute error = {}".format(round(self.mean_absolute_error,round_amount)))
        print("Mean squared error = {}".format(round(self.mean_squared_error,round_amount)))
        print("Root mean squared error = {}".format(round(self.root_mean_squared_error,round_amount)))
        print("Standard deviation error = {}".format(round(self.std_error,round_amount)))
        print("Correlation = {}".format(round(self.correlation,round_amount)))
        print("+95% Confidence Interval = {}".format(round(self.CI95[0],round_amount)))
        print("-95% Confidence Interval = {}".format(round(self.CI95[1],round_amount)))

    def return_stats(self):
        # return dict of statistics
        stats_dict = {'mean_error': self.mean_error,
        'mean_absolute_error':self.mean_absolute_error,
        'mean_squared_error':self.mean_squared_error,
        'root_mean_squared_error':self.root_mean_squared_error,
        'correlation':self.correlation,
        'CI_95%+':self.CI95[0],
        'CI_95%-':self.CI95[1]}

        return stats_dict

    def rand_jitter(self, arr):
        stdev = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def scatter_plot(self,x_label='Gold Standard',y_label='New Measure',
                    figure_size=(4,4), show_legend=True,
                    the_title=' ',
                    file_name='BlandAltman_ScatterPlot.pdf',
                    is_journal=False, measure_lower_lim=40, measure_upper_lim=150):

        if is_journal: # avoid use of type 3 fonts for journal paper acceptance
            import matplotlib
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

        font_size = 18
        # set the font to Charter
        font = {'family': 'serif', 'serif': ['Charter'], 'size': font_size}
        plt.rc('font', **font)
        # plt.rc('xtick', labelsize=font_size)
        # plt.rc('ytick', labelsize=font_size)

        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        self.gold_std = self.rand_jitter(self.gold_std)
        self.new_measure = self.rand_jitter(self.new_measure)

        fig = plt.figure(figsize=figure_size)
        ax=fig.add_axes([0,0,1,1])
        xy = np.vstack([self.gold_std,self.new_measure])
        z = gaussian_kde(xy)(xy)
        ax.scatter(self.gold_std,self.new_measure, c=z, s=50)
        x_vals = np.array(ax.get_xlim())
        ax.plot(x_vals,x_vals,'--',color='black', label='Line of Slope = 1')

        # Plot the identity line with y-value offset by +2 and -2
        offset = 2
        ax.plot([min(x_vals), max(x_vals)], [min(x_vals) + offset, max(x_vals) + offset],
                 color='red', linestyle='--', label='Identity Line +2 BPM', alpha=0.6)

        offset = -2
        ax.plot([min(x_vals), max(x_vals)], [min(x_vals) + offset, max(x_vals) + offset],
                 color='red', linestyle='--', label='Identity Line -2 BPM', alpha=0.6)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.set_title(the_title)
        ax.grid()
        plt.xlim(measure_lower_lim, measure_upper_lim)
        plt.ylim(measure_lower_lim, measure_upper_lim)
        plt.savefig(os.path.join(self.save_path, file_name),bbox_inches='tight', dpi=300)
        print(f"Saved {file_name} to {self.save_path}.")

    def difference_plot(self,x_label='Difference between rPPG HR and ECG HR [bpm]',
                        y_label='Average of rPPG HR and ECG HR [bpm]',averaged=False,
                        figure_size=(4,4),show_legend=True,
                        the_title='',file_name='BlandAltman_DifferencePlot.pdf',
                        is_journal=False):

        if is_journal: # avoid use of type 3 fonts for journal paper acceptance
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

        font_size = 18
        # set the font to Charter
        font = {'family': 'serif', 'serif': ['Charter'], 'size': font_size}
        plt.rc('font', **font)
        # plt.rc('xtick', labelsize=font_size)
        # plt.rc('ytick', labelsize=font_size)

        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        diffs = self.gold_std - self.new_measure
        avgs = (self.gold_std + self.new_measure) / 2

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_axes([0,0,1,1])
        xy = np.vstack([avgs,diffs])
        z = gaussian_kde(xy)(xy)
        ax.scatter(avgs,diffs, c=z, label='Beobachtungen')
        x_vals = np.array(ax.get_xlim())
        ax.axhline(self.mean_error,color='black',label='Mittlerer Fehler')  # 'Mean Error')
        ax.axhline(self.CI95[0],color='black',linestyle='--',label='+95% Konfidenzintervall')
        ax.axhline(self.CI95[1],color='black',linestyle='--',label='-95% Konfidenzintervall')
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)
        # ax.set_title(the_title)
        if show_legend:
            ax.legend(loc='upper left')
        ax.grid()
        plt.savefig(os.path.join(self.save_path, file_name),bbox_inches='tight', dpi=100)
        print(f"Saved {file_name} to {self.save_path}.")
