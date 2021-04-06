from utils import *
import matplotlib.pyplot as plt

path_save = "C:/Users/benya/Desktop/"

path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/2_QQE/datasets/three_blobs/dimension_2/"
itr = "initial"
X = load_variable(name_of_variable="X", path=path_)
X = X.T
y = load_variable(name_of_variable="y", path=path_)
color_map = plt.cm.brg  #--> plt.cm.brg, plt.cm.hsv

plt.scatter(X[0, :], X[1, :], c=y, cmap=color_map, edgecolors='k')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.show()
plt.savefig(path_save + str(itr) + ".png")
plt.clf()
plt.close()

X_separated_classes, indices_of_points_in_classes = separate_samples_of_classes_2(X=X, y=y)

color_map = plt.cm.brg  #--> plt.cm.brg, plt.cm.hsv

# color = []
X_multiclass_to_plot = X.copy()
for class_index in range(3):
    path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/9_synthetic_separationOfClasses/algorithm_files/class_" + str(class_index) + "/fuzzy_QQplot/iterations_numpy/"
    itr = 500
    name = "X_matched_iteration_"+str(itr)
    X_class = load_variable(name_of_variable=name, path=path_)
    X_multiclass_to_plot[:, indices_of_points_in_classes[class_index]] = X_class
    # color.extend([class_index] * X_class.shape[1])

    plt.scatter(X_multiclass_to_plot[0, :], X_multiclass_to_plot[1, :], c=y, cmap=color_map, edgecolors='k')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()
    plt.savefig(path_save + "class" + str(class_index) + "_itr" + str(itr) + ".png")
    plt.clf()
    plt.close()







