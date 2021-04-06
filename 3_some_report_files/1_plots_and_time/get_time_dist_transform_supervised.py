from utils import *


# path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/9_synthetic_separationOfClasses/"
# n_classes = 3

path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/17_face_glasses_transform_inputSpace/run2_good/"
n_classes = 2

time_matching_list = []
time_quasi_newton_list = []
for class_index in range(n_classes):
    # print("class_index: " + str(class_index))
    path_file = path_ + "algorithm_files/class_" + str(class_index) + "/matched_data/"
    time_matching = load_variable(name_of_variable="time_matching", path=path_file)
    time_matching_list.append(time_matching)
    path_file = path_ + "algorithm_files/class_" + str(class_index) + "/fuzzy_QQplot/"
    time_quasi_newton = load_variable(name_of_variable="time_quasi_newton", path=path_file)
    time_quasi_newton_list.append(time_quasi_newton[-1])
total_time = sum(time_matching_list) + sum(time_quasi_newton_list)
print(total_time)

