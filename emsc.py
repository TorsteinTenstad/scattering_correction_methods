from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
import utility


def binary_case_regression(lambdas, pure_spectrum_a, pure_spectrum_b, scattered_spectrum, maximum_wavelength_depencency_order):
    ones = np.ones_like(lambdas)
    m = (pure_spectrum_a+pure_spectrum_b)/2
    k = pure_spectrum_a-pure_spectrum_b
    lambda_n = np.array([np.power(lambdas, i+1) for i in range(maximum_wavelength_depencency_order)])
    M_t = np.vstack((ones, m, k, lambda_n))
    M = M_t.T
    
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(M, scattered_spectrum)
    
    return tuple(regr.coef_)


def correct_spectra(frequencies, spectra, a, b, wave_coefficients):
    return (spectra - a - np.sum(np.array([coefficient*np.power(frequencies, i+1) for i, coefficient in enumerate(wave_coefficients)]), axis=0))/b


def binary_emsc(lambdas, scattered_spectum, pure_spectrum_a, pure_spectrum_b):
    a, b, h, *wave_coefficients = binary_case_regression(lambdas, pure_spectrum_a, pure_spectrum_b, scattered_spectum, 2)
    corrected_spectrum = correct_spectra(lambdas, scattered_spectum, a, b, np.array(wave_coefficients))
    return corrected_spectrum, h


def emsc(lambdas, scattered_data, pure_spectrum_a, pure_spectrum_b):
    return utility.apply_func_on_set(scattered_data, lambda spectrum : binary_emsc(lambdas, spectrum, pure_spectrum_a, pure_spectrum_b)[0])