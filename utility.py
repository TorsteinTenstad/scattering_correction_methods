def apply_func_on_set(set, func):
    output = [None]*len(set)
    for c, spectra_c in enumerate(set):
        output[c] = [None]*len(spectra_c)
        for i, spectrum, in enumerate(spectra_c):
            output[c][i] = func(spectrum)
    return output