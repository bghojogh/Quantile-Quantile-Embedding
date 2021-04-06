from utils import *


path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes2_after_revision/7_hyperparams_results/"

for parameter_ in ["lambda", "k", "eta"]:
    if parameter_ == "lambda":
        list_ = ["lambda=0x1", "lambda=1", "lambda=10"]
    elif parameter_ == "k":
        list_ = ["k=1", "k=50"]
    elif parameter_ == "eta":
        list_ = ["eta=0x1", "eta=1", "eta=10"]
    with open("./times"+'.txt', 'a+') as file:
        file.write("=====================================\n\n")
    for exact_or_shape in ["shape_S_to_uniform", "exact_S_to_uniform"]:
        path_ = path_base + (parameter_ + "/" + exact_or_shape + "/")
        for param in list_:
            print("-----")
            print("param: " + param)
            path_file = path_ + param + "/algorithm_files/fuzzy_QQplot/"
            time_quasi_newton = load_variable(name_of_variable="time_quasi_newton", path=path_file)
            time_of_iterations = [time_quasi_newton[i] if i == 0 else time_quasi_newton[i] - time_quasi_newton[i-1] for i in range(len(time_quasi_newton))]
            str_ = "{}, {}: mean time = {} \n\n".format( exact_or_shape, param, np.mean(np.asarray(time_of_iterations)) )
            print(str_)
            with open("./times"+'.txt', 'a+') as file:
                file.write(str_)

