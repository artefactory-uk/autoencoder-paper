import numpy as np
import scipy.stats


class ConfidenceIntervals:
    def __init__(self, CI: float = 0.95):
        self.CI = CI

    @staticmethod
    def __calculate_confidence_interval(
        curve_points: list, CI: float = 0.95
    ) -> (float, float, float):
        """
        Calculates CI for one epoch in the learning curve.
        """
        points = np.array(curve_points)
        num_points = len(points)
        mean, standard_error = np.mean(points), scipy.stats.sem(points)
        lower_bound, upper_bound = scipy.stats.t.interval(
            CI, num_points - 1, loc=mean, scale=standard_error
        )
        return mean, lower_bound, upper_bound

    def get_intervals(self, epochs_dict: dict) -> (list, list, list):
        all_means, all_upper, all_lower = [], [], []
        for epoch in epochs_dict.keys():
            mean, upper, lower = self.__calculate_confidence_interval(
                epochs_dict[epoch], CI=self.CI
            )
            all_means.append(mean), all_upper.append(upper), all_lower.append(lower)
        return all_lower, all_means, all_upper
