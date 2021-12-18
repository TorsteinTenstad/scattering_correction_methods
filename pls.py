import numpy as np
from sklearn.cross_decomposition import PLSRegression


def pls(train_set, test_set, concentrations, pls_components):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    test_y_indices = []
    for i, concentration in enumerate(concentrations):
        for spectrum in train_set[i]:
            train_x.append(spectrum)
            train_y.append(concentration)
        for spectrum in test_set[i]:
            test_x.append(spectrum)
            test_y.append(concentration)
            test_y_indices.append(i)

    pls2 = PLSRegression(pls_components)
    pls2.fit(train_x, train_y)

    pred_y = pls2.predict(test_x)[:,0]

    predicted = {i: [] for i in range(len(concentrations))}
    for true_i, pred in zip(test_y_indices, pred_y):
        predicted[true_i].append(pred)

    return predicted


def pls_mse(concentrations, pls_result):
    per_concentration_mse=[np.average((pred-concentrations[true_i])**2) for true_i, pred in pls_result.items()]
    return np.average(per_concentration_mse), per_concentration_mse