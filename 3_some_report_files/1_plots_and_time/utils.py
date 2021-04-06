import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

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

def scatter_of_data_3(X, y, plot_name, path_save_plot, color_map):
    # X --> rows: features, columns: samples
    X_separated_classes, indices_of_points_in_classes = separate_samples_of_classes_2(X=X, y=y)
    n_classes = len(X_separated_classes)
    for class_index in range(n_classes):
        color_meshgrid = np.ones((X_separated_classes[class_index].shape[1],)) * class_index
    X_multiclass_to_plot = np.zeros(X.shape)
    n_samples = X.shape[1]
    labels_of_kept_points = np.zeros((n_samples,))
    for class_index in range(n_classes):
        X_multiclass_to_plot = np.column_stack((X_multiclass_to_plot, X_separated_classes[class_index]))
    plt.scatter(X_multiclass_to_plot[0, :], X_multiclass_to_plot[1, :], c=color_meshgrid, cmap=color_map, edgecolors='k')
    # plt.show()
    plt.savefig(path_save_plot + plot_name + ".png")
    plt.clf()
    # plt.close()