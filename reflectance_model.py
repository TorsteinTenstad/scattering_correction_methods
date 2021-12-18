import numpy as np
from newtons_method import newtons_method
import utility

def gamma(A, musr, mua):
    gamma_ = A*musr/((1+A)*mua+A*musr+(1/3+A)*np.sqrt(3*musr*mua+3*mua**2))
    return gamma_


def D_gamma(A, musr, mua):
    gamma_ = gamma(A, musr, mua)
    D_gamma_ = -gamma_**2/(A*musr)*((1+A)+((A+1/3)*(3*musr+6*mua))/(2*np.sqrt(3*musr*mua+3*mua**2)))
    return D_gamma_


def fit_gamma_model(A, musr, spectrum, mua_0=None):
    mua = mua_0 if mua_0 else 20*np.ones(len(spectrum))
    gamma_ = np.empty(len(spectrum))

    for i, x in enumerate(spectrum):
        f = lambda mua : gamma(A, musr[i], mua) - spectrum[i]
        D_f = lambda mua : D_gamma(A, musr[i], mua)
        mua[i] = newtons_method(f, D_f, mua[i], epsilon=1e-4, minimum_defined_value=0.001)[0]
        gamma_[i] = gamma(A, musr[i], mua[i])
    return mua


def estimate_mua(A, musr, data, mua_0=None):
    return utility.apply_func_on_set(data, lambda spectrum : fit_gamma_model(A, musr, spectrum, mua_0))