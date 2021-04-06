from utils import *


path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/15_MNIST_unsupervised/"
initial_methods = ["PCA", "FDA", "Isomap", "LLE", "TSNE", "resnet_crossEntropy", "resnet_Siamese"]
for method in initial_methods:
    print("-----")
    print("method: " + method)
    path_file = path_ + method + "/algorithm_files" + "/matched_data/"
    time_matching = load_variable(name_of_variable="time_matching", path=path_file)
    path_file = path_ + method + "/algorithm_files" + "/fuzzy_QQplot/"
    time_quasi_newton = load_variable(name_of_variable="time_quasi_newton", path=path_file)
    total_time = time_matching + time_quasi_newton[-1]
    print(total_time)

