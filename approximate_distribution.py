import numpy as np

def approximate_distribution(intervals, p_intervals, edge_cases):
    m = 1000
    r = np.append([], [np.linspace(interval[0], interval[1], m) for interval in intervals])
    pr = np.append([], [p*np.ones(m)/m for p in p_intervals])
    r = np.append(r, edge_cases[0])
    pr = np.append(pr, edge_cases[1])
    return r, pr