import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import rc
import pickle


from src.paths import CI_EXPERIMENT_PATH

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Founders Grotesk']})
plt.rcParams.update({'font.size': 29})

class ConfidenceIntervals():
    def __init__(self, CI = 0.95):
        self.CI = CI


    def __calculate_CI(self,curve_points, CI = 0.95):
        '''
        Calculates CI for one epoch in the learning curve.
        '''
        points = np.array(curve_points)
        num_points = len(points)
        mean, standard_error = np.mean(points), scipy.stats.sem(points)
        lower_bound, upper_bound = scipy.stats.t.interval(CI, num_points -1, loc = mean, scale = standard_error)
        return mean, lower_bound, upper_bound

    def get_intervals(self, epochs_key):
        all_means, all_upper, all_lower = [], [], []
        for epoch in epochs_key.keys():
            mean, upper, lower = self.__calculate_CI(epochs_key[epoch], CI=self.CI)
            all_means.append(mean), all_upper.append(upper), all_lower.append(lower)
        return all_lower, all_means, all_upper
