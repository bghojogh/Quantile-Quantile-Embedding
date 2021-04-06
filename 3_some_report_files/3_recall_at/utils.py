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

def evaluate_embedding(embedding, labels, path_save_accuracy_of_test_data, k_list=[1, 2, 4, 8, 16], name="temp"):
    # https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/train.py
    """Evaluate embeddings based on Recall@k."""
    d_mat = get_distance_matrix(embedding)
    # d_mat = d_mat.asnumpy()
    # labels = labels.asnumpy()
    # le = preprocessing.LabelEncoder()
    # le.fit(np.unique(np.asarray(labels)))
    # labels = le.transform(labels)
    recall_at = []
    for k in k_list:
        print('Recall@%d' % k)
        correct, cnt = 0.0, 0.0
        for i in range(embedding.shape[0]):
            d_mat[i, i] = np.inf
            # https://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output
            nns = np.argpartition(d_mat[i], k)[:k]   
            if any(labels[i] == labels[nn] for nn in nns):
                correct += 1
            cnt += 1
        recall_at.append(correct/cnt)
    k_list = np.asarray(k_list)
    recall_at = np.asarray(recall_at)
    # save results:
    path_ = path_save_accuracy_of_test_data + "recall_at\\"
    if not os.path.exists(path_):
        os.makedirs(path_)
    np.save(path_+"k_list_"+name+".npy", k_list)
    np.savetxt(path_+"k_list_"+name+".txt", k_list, delimiter=',')   
    np.save(path_+"recall_at_"+name+".npy", recall_at)
    np.savetxt(path_+"recall_at_"+name+".txt", recall_at, delimiter=',')   
    return k_list, recall_at

def get_distance_matrix(x):
    # https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/train.py
    """Get distance matrix given a matrix. Used in testing."""
    square = np.sum(x ** 2.0, axis=1).reshape((-1, 1))
    distance_square = square + square.transpose() - (2.0 * np.dot(x, x.transpose()))
    distance_square[distance_square < 0] = 0
    return np.sqrt(distance_square)