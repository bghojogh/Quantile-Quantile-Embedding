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

# experiment = "5_S_2D"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/run3_good/algorithm_files/"

# experiment = "6_S_2D_swap"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/algorithm_files/"

# experiment = "7_S_2D_Exact"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/algorithm_files/"

# experiment = "8_S_2D_Exact_swap"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/algorithm_files/"

# experiment = "11_3blobs_3D_Unsupervised"
# initialization_method = "TSNE"  # PCA, FDA, Isomap, LLE, TSNE
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/algorithm_files/"
# # path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/1_scaledLLE_good/algorithm_files/"

# experiment = "15_MNIST_unsupervised"
# initialization_method = "resnet_Siamese"  # PCA, FDA, Isomap, LLE, TSNE, resnet_crossEntropy, resnet_Siamese
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/" + initialization_method + "/algorithm_files/"

# experiment = "16_Face_glasses_distributionTransform"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/algorithm_files/"
# apply_pca = True

# experiment = "18_CDF"
# path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/" + experiment + "/algorithm_files/"

experiment = "19_histopathology_unsupervised"
path_base = "C:/Users/benya/Desktop/my_PhD/QQE/codes2_after_revision/9_histopathology_results/FDT_loss/Shape_Gaussian/unsupervised_Gaussian/algorithm_files/"

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
        if itr == "first":
            path_ = path_base + "/fuzzy_QQplot/"
            name = "X_matched_initial"
        else:
            path_ = path_base + "/fuzzy_QQplot/iterations_numpy/"
            # last_itr = max([i.split("\\")[-1].split(".pckl")[0].split("_")[-1] for i in glob.glob(path_+"*")])
            last_itr = max([i.split("\\")[-1].split(".pckl")[0].split("_")[-1] for index_, i in enumerate(glob.glob(path_+"*")) if index_ % 3 == 0])
            name = "X_matched_iteration_"+str(last_itr)
            name = "X_matched_iteration_"+str(last_itr)
        X = load_variable(name_of_variable=name, path=path_)
        path_ = path_base
        name = "reference_sample"
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
        measure_mean = measure
        print("{}: {}\n".format(measure_method, measure_mean))
        str_ += "{}: {}\n\n".format(measure_method, measure_mean)
    with open(path_save+experiment+"_"+initialization_method+'.txt', 'a+') as file:
        file.write(str_)


if __name__ == "__main__":
    main()