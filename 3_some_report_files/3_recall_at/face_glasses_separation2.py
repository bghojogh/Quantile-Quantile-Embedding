from utils import *
import matplotlib.pyplot as plt
# import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from PIL import Image


path_save = "./results/face_glasses_separation2/"

if not os.path.exists(path_save):
    os.makedirs(path_save)

# color_map =  matplotlib.colors.hsv_to_rgb(plt.cm.hsv) # plt.cm.bwr  #--> plt.cm.brg, plt.cm.hsv
# color_map = plt.cm.bwr

path_1 = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/17_face_glasses_transform_inputSpace/run2_good/algorithm_files/class_" + str(0) + "/fuzzy_QQplot/"
X0 = load_variable(name_of_variable="X_matched_initial", path=path_1)
path_1 = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/17_face_glasses_transform_inputSpace/run2_good/algorithm_files/class_" + str(1) + "/fuzzy_QQplot/"
X1 = load_variable(name_of_variable="X_matched_initial", path=path_1)
X = np.column_stack((X0, X1))
y = [0]*X0.shape[1] + [1]*X1.shape[1] 
y = np.asarray(y)

for i, plot_name in enumerate(["X_matched_iteration_0", "X_matched_iteration_20", "X_matched_iteration_30", "X_matched_iteration_10"]):
    if i <= 2:
        class_index_of_plot = 0
    else:
        class_index_of_plot = 1
    path_1 = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/17_face_glasses_transform_inputSpace/run2_good/algorithm_files/class_" + str(class_index_of_plot) + "/fuzzy_QQplot/iterations_numpy/"
    X_class = load_variable(name_of_variable=plot_name, path=path_1)
    if i != 0:
        X[:, y==class_index_of_plot] = X_class
    
    # plt.scatter(X[0, :], X[1, :], c=y, cmap=color_map, edgecolors='k')
    markers = ["v", "o"]
    colors = ["r", "b"]
    for class_index in range(2):
        sample_of_this_class = X[:, y == class_index]
        # c = class_index * np.ones((sample_of_this_class.shape[1],))
        plt.scatter(sample_of_this_class[0, :], sample_of_this_class[1, :], s=30, color=colors[class_index], alpha=1.0, marker=markers[class_index])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()
    plt.savefig(path_save + str(i) + ".png")
    plt.clf()
    plt.close()

    evaluate_embedding(embedding=X.T, labels=y, path_save_accuracy_of_test_data=path_save, k_list=[1, 2, 4, 8, 16], name=str(i))
