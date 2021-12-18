import numpy as np
import pickle
from matplotlib import pyplot as plt


def data_preparation():
    from data_preparation import read_and_calibrate, average_over_regions, random_equal_split
    lambdas, R_data, concentrations = read_and_calibrate()
    samples = average_over_regions(R_data, region_size=(15, 15))
    R_training, R_test = random_equal_split(samples)

    raw_A_training = [-np.log10(samples) for samples in R_training]
    raw_A_test = [-np.log10(samples) for samples in R_test]
    return lambdas, R_training, R_test, raw_A_training, raw_A_test, concentrations


def correction(lambdas, R_training, R_test, raw_A_training, raw_A_test):
    # EMSC:
    from emsc import emsc
    pure_sand_apparent_absorpance = np.average(raw_A_test[0], axis=0)
    pure_limestone_apparent_absorpance = np.average(raw_A_test[-1], axis=0)

    EMSC_A_training = emsc(lambdas, raw_A_training, pure_sand_apparent_absorpance, pure_limestone_apparent_absorpance)
    EMSC_A_test = emsc(lambdas, raw_A_test, pure_sand_apparent_absorpance, pure_limestone_apparent_absorpance)

    # Optical model:
    from a import a
    from approximate_distribution import approximate_distribution
    from mie import get_musr
    from reflectance_model import estimate_mua

    n_s = 1.44
    n_b = 1
    A = a(n_s, n_b)

    intervals = 1e-3*np.array([[0.07, 0.1], [0.1, 0.15], [0.15, 0.42], [0.42, 0.6], [0.6, 0.8]])/2
    p_intervals = 1e-2*np.array([4.8, 25, 57, 8, 3.5])
    edge_cases=[[0.07, 0.8], [0.015, 0.002]]
    r, pr = approximate_distribution(intervals, p_intervals, edge_cases)

    musr = get_musr(lambdas, r, pr, n_s, n_b)
    mua_training = estimate_mua(A, musr, R_training)
    mua_test = estimate_mua(A, musr, R_test)
    data_collections = [[raw_A_training, raw_A_test], [EMSC_A_training, EMSC_A_test], [mua_training, mua_test]]
    return data_collections


def pls_evalation(data_collections, concentrations):
    from pls import pls
    pls_results = [pls(*collection, concentrations, pls_components=2) for collection in data_collections]
    return pls_results


if __name__ == "__main__":
    results = {}

    plot_only = False
    if not plot_only:
        lambdas, R_training, R_test, raw_A_training, raw_A_test, concentrations = data_preparation()
        data_collections = correction(lambdas, R_training, R_test, raw_A_training, raw_A_test)
        pls_results = pls_evalation(data_collections, concentrations)

        results = {'lambdas': lambdas,
                'concentrations': concentrations,
                'data_collections': data_collections,
                'pls_results': pls_results}
        with open('results.pickle', 'wb') as output_file:
            pickle.dump(results, output_file)
    else:
        with open('results.pickle', 'rb') as input_file:
            results = pickle.load(input_file)
        lambdas = results['lambdas']
        concentrations = results['concentrations']
        data_collections = results['data_collections']
        pls_results = results['pls_results']
        
    from plotting import plot_spectra, plot_pls, single_sample_comparison
    plot_spectra(lambdas, concentrations, data_collections)
    plot_pls(concentrations, pls_results)
    single_sample_comparison(lambdas, concentrations, data_collections, 0)
    plt.show()