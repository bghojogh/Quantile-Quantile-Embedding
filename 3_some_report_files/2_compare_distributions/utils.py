import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import stats


def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def remove_outliers(data_, color_meshgrid):
    # data_: column-wise samples
    data_outliers_removed = data_.copy()
    color_meshgrid_outliers_removed = color_meshgrid.copy()
    for dimension_index in range(data_.shape[0]):
        data_dimension = data_[dimension_index, :].ravel()
        # Set upper and lower limit to 3 standard deviation
        data_dimension_std = np.std(data_dimension)
        data_dimension_mean = np.mean(data_dimension)
        anomaly_cut_off = data_dimension_std * 3
        lower_limit = data_dimension_mean - anomaly_cut_off
        upper_limit = data_dimension_mean + anomaly_cut_off
        samples_to_keep = []
        for sample_index in range(data_outliers_removed.shape[1]):
            sample_ = data_outliers_removed[:, sample_index]
            if sample_[dimension_index] > upper_limit or sample_[dimension_index] < lower_limit:
                samples_to_keep.append(False)
            else:
                samples_to_keep.append(True)
        data_outliers_removed = data_outliers_removed.compress(samples_to_keep, axis=1)
        color_meshgrid_outliers_removed = color_meshgrid_outliers_removed.compress(samples_to_keep)
    return data_outliers_removed, color_meshgrid_outliers_removed

def separate_samples_of_classes_2(X, y):  # it does not change the order of the samples within every class
    # X --> rows: features, columns: samples
    # return X_separated_classes --> each element of list --> rows: features, columns: samples
    y = np.asarray(y)
    y = y.reshape((-1, 1)).ravel()
    labels_of_classes = sorted(set(y.ravel().tolist()))
    n_samples = X.shape[1]
    n_dimensions = X.shape[0]
    n_classes = len(labels_of_classes)
    X_separated_classes = [np.empty((n_dimensions, 0))] * n_classes
    original_index_in_whole_dataset = [None] * n_classes
    for class_index in range(n_classes):
        original_index_in_whole_dataset[class_index] = []
        for sample_index in range(n_samples):
            if y[sample_index] == labels_of_classes[class_index]:
                X_separated_classes[class_index] = np.column_stack((X_separated_classes[class_index], X[:, sample_index].reshape((-1,1))))
                original_index_in_whole_dataset[class_index].append(sample_index)
    return X_separated_classes, original_index_in_whole_dataset

def HSIC(X, Y, kernel_type="rbf"):
    # X and Y: rows are features and columns are samples
    assert X.shape[1] == Y.shape[1]
    K_x = pairwise_kernels(X=X.T, Y=X.T, metric=kernel_type)
    K_y = pairwise_kernels(X=Y.T, Y=Y.T, metric=kernel_type)
    # K_x = normalize_the_kernel(K_x)
    # K_y = normalize_the_kernel(K_y)
    K_y = center_the_matrix(K_y, mode="double_center")
    n_samples = K_x.shape[0]
    HSIC_ = np.trace(K_x @ K_y) * (1 / (n_samples-1)**2)
    return HSIC_

def MMD(X, Y, kernel_type="rbf"):
    # X and Y: rows are features and columns are samples
    K_x = pairwise_kernels(X=X.T, Y=X.T, metric=kernel_type)
    K_y = pairwise_kernels(X=Y.T, Y=Y.T, metric=kernel_type)
    K_x_y = pairwise_kernels(X=X.T, Y=Y.T, metric=kernel_type)
    # K_x = normalize_the_kernel(K_x)
    # K_y = normalize_the_kernel(K_y)
    # K_x_y = normalize_the_kernel(K_x_y)
    n_samples_x = K_x.shape[0]
    n_samples_y = K_y.shape[0]
    term1 = np.sum(K_x) * (1 / n_samples_x**2)  #--> or (equaivalent to): np.mean(K_x)
    term2 = np.sum(K_y) * (1 / n_samples_y**2)  #--> or (equaivalent to): np.mean(K_y)
    term3 = np.sum(K_x_y) * (-2 / (n_samples_x*n_samples_y))  #--> or (equaivalent to): -2 * np.mean(K_x_y)
    MMD_ = term1 + term2 + term3
    return MMD_

def KL(X, Y):
    kernel = stats.gaussian_kde(X)  #--> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    pdf_x = kernel.pdf(X)
    pdf_x /= sum(pdf_x)
    kernel = stats.gaussian_kde(Y)  
    pdf_y = kernel.pdf(Y)
    pdf_y /= sum(pdf_y)
    KL_ = compute_kl_divergence(pdf_x, pdf_y)
    return KL_

def compute_kl_divergence(p_probs, q_probs):
    # http://ethen8181.github.io/machine-learning/model_selection/kl_divergence.html
    """"KL (p || q)"""
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div)

def center_the_matrix(the_matrix, mode="double_center"):
    n_rows = the_matrix.shape[0]
    n_cols = the_matrix.shape[1]
    vector_one_left = np.ones((n_rows,1))
    vector_one_right = np.ones((n_cols, 1))
    H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
    H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
    if mode == "double_center":
        the_matrix = H_left.dot(the_matrix).dot(H_right)
    elif mode == "remove_mean_of_rows_from_rows":
        the_matrix = H_left.dot(the_matrix)
    elif mode == "remove_mean_of_columns_from_columns":
        the_matrix = the_matrix.dot(H_right)
    return the_matrix

def normalize_the_kernel(kernel_matrix):
    diag_kernel = np.diag(kernel_matrix)
    # print(diag_kernel)
    # input("hi")
    k = (1 / np.sqrt(diag_kernel)).reshape((-1,1))
    normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
    return normalized_kernel_matrix