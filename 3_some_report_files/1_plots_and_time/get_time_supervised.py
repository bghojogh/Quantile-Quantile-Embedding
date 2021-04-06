from utils import *


path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/14_MNIST_supervised/"
initial_methods = ["PCA", "FDA", "Isomap", "LLE", "TSNE", "resnet_crossEntropy", "resnet_Siamese"]

# path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/13_MNIST_Exact_supervised/"
# initial_methods = ["MNIST_1000"]

n_classes = 10

for method in initial_methods:
    print("-----")
    print("method: " + method)
    time_matching_list = []
    time_quasi_newton_list = []
    for class_index in range(n_classes):
        # print("class_index: " + str(class_index))
        path_file = path_ + method + "/algorithm_files/class_" + str(class_index) + "/matched_data/"
        time_matching = load_variable(name_of_variable="time_matching", path=path_file)
        time_matching_list.append(time_matching)
        path_file = path_ + method + "/algorithm_files/class_" + str(class_index) + "/fuzzy_QQplot/"
        time_quasi_newton = load_variable(name_of_variable="time_quasi_newton", path=path_file)
        time_quasi_newton_list.append(time_quasi_newton[-1])
    total_time = sum(time_matching_list) + sum(time_quasi_newton_list)
    print(total_time)

