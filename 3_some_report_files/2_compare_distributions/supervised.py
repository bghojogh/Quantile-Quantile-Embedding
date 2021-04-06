from utils import *
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA


# path_save = "C:/Users/benya/Desktop/"
path_save = "./results/"

kernel_type = "rbf" #--> "rbf", "sigmoid", "polynomial", "poly", "linear", "cosine"
initialization_method = ""
apply_pca = False

############ Uncomment the blocks of experiments one by one:

# experiment = "9_synthetic_separationOfClasses"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/algorithm_files/"
# n_classes = 3

# experiment = "10_3blobs_3D_Supervised"
# initialization_method = "TSNE"  # PCA, FDA, Isomap, LLE, TSNE
# # path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/run2_good/algorithm_files/"
# # path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/run2_better/algorithm_files/"
# # path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/run3_scaledLLE_good/algorithm_files/"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/algorithm_files/"
# n_classes = 3

# experiment = "12_3blobs_3D_Exact"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/PCA/run2_good/algorithm_files/"
# n_classes = 3

# experiment = "13_MNIST_Exact_supervised"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/MNIST_1000/algorithm_files/"
# n_classes = 10

# experiment = "14_MNIST_supervised"
# initialization_method = "resnet_Siamese"  # PCA, FDA, Isomap, LLE, TSNE, resnet_crossEntropy, resnet_Siamese
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/algorithm_files/"
# n_classes = 10

# experiment = "17_face_glasses_transform_inputSpace"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/run2_good/algorithm_files/"
# n_classes = 2
# apply_pca = True

# experiment = "20_histopathology_supervised"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes2_after_revision/9_histopathology_results/FDT_loss/Shape_Gaussian/supervised_Gaussian/algorithm_files/"
# n_classes = 8

experiment = "21_histopathology_exact"
path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes2_after_revision/9_histopathology_results/FDT_loss/Exact_Gaussian/supervised/algorithm_files/"
n_classes = 8

############ 

def main():
    calculate("first")
    calculate("last")

def calculate(itr="last"):
    X = np.empty((2, 0))
    Y = np.empty((2, 0))
    if itr == "first":
        str_ = "=========================\n itr: {}\n\n".format(0)
    else:
        str_ = "=========================\n itr: {}\n\n".format("last")
    print(str_)
    for measure_method in ["KL", "MMD", "HSIC"]:
        measure_mean = 0
        for class_index in range(n_classes):
            if itr == "first":
                path_ = path_base + "class_" + str(class_index) + "/fuzzy_QQplot/"
                name = "X_matched_initial"
            else:
                path_ = path_base + "class_" + str(class_index) + "/fuzzy_QQplot/iterations_numpy/"
                # last_itr = max([i.split("\\")[-1].split(".pckl")[0].split("_")[-1] for i in glob.glob(path_+"*")])
                last_itr = max([i.split("\\")[-1].split(".pckl")[0].split("_")[-1] for index_, i in enumerate(glob.glob(path_+"*")) if index_ % 3 == 0])
                name = "X_matched_iteration_"+str(last_itr)
            X = load_variable(name_of_variable=name, path=path_)
            path_ = path_base + "class_" + str(class_index) + "/matched_data/"
            name = "Y_matched"
            Y = load_variable(name_of_variable=name, path=path_)
            if X.shape[1] != Y.shape[1]:
                X_ = X[:, :min(X.shape[1], Y.shape[1])-1]
                Y_ = Y[:, :min(X.shape[1], Y.shape[1])-1]
                X, Y = X_, Y_
            if apply_pca:
                pca = PCA(n_components=50)
                X = pca.fit_transform(X.T).T
                pca = PCA(n_components=50)
                Y = pca.fit_transform(Y.T).T
            ### measure:
            if measure_method == "HSIC":
                measure = HSIC(X, Y, kernel_type)
            elif measure_method == "MMD":
                measure = MMD(X, Y, kernel_type)
            elif measure_method == "KL":
                measure = KL(X, Y)
            print("{} of class{}: {}".format(measure_method, str(class_index), measure))
            str_ += "{} of class{}: {}\n".format(measure_method, str(class_index), measure)
            measure_mean += measure
        measure_mean /= n_classes
        print("mean {}: {}\n".format(measure_method, measure_mean))
        str_ += "mean {}: {}\n\n".format(measure_method, measure_mean)
    with open(path_save+experiment+"_"+initialization_method+'.txt', 'a+') as file:
        file.write(str_)


if __name__ == "__main__":
    main()