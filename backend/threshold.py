import pandas as pd
import numpy as np
from math import e

def histogram_analysis(close_diff, number_of_bins):
    """
    Calculates an upper threshold for EUR/USD daily differences based on distribution

    The differences are placed into 10 equally sized bins
    Bin counts are sorted and are cumulatively summed until it exceeds 85% of the dataset size
    The maximum value of the last contributing bin is returned as the threshold
    """
    close_diff = close_diff.abs()
    bin_values = pd.cut(close_diff, bins=number_of_bins)
    bin_counts = close_diff.groupby(bin_values).sum()
    sum_bin_counts = bin_counts.sum()
    temp_sum = i = 0
    while i < number_of_bins:
        temp_sum += bin_counts.iloc[i]

        if temp_sum/sum_bin_counts > 0.85:
            break
        i += 1
    return bin_values.cat.categories[i].right

def calculate_entropy(labels, base = e):
  """
  Calculate the degree of randomness in the labels
  """
  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
  return -(vc * np.log(vc)/np.log(base)).sum()

def calculate_threshold(close_diff):
    """
    Finds the threshold that maximises entropy

    Thresholds are tested in increments of 0.00001 between 0 and threshold_upper_bound
    For each threshold, changes are classified as:
        - 2: increase (above threshold)
        - 1: decrease (below negative threshold)
        - 0: no-change (within threshold range)

    The entropy of these classes is calculated for each threshold, and the threshold with the highest entropy is selected
    """
    threshold_upper_bound = histogram_analysis(close_diff, 10)
    print(f"Threshold upper bound:{threshold_upper_bound}")
    temp_threshold = threshold = 0
    best_entropy = -float("infinity")
    close_diff = close_diff.tolist()
    
    while temp_threshold < threshold_upper_bound:
        labels = [2 if x > temp_threshold else 1 if x < -temp_threshold else 0 for x in close_diff]
        entropy = calculate_entropy(labels)

        if entropy > best_entropy:
            best_entropy = entropy
            threshold = temp_threshold
            print(f"Best threshold = {temp_threshold}, entropy = {entropy}")
        temp_threshold += 0.00001

    return threshold