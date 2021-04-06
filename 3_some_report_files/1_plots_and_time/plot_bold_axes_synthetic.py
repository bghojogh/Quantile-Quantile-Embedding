from utils import *
import matplotlib.pyplot as plt

path_save = "C:/Users/benya/Desktop/"

X = np.empty((2, 0))
color = []
for class_index in range(3):
    path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/12_3blobs_3D_Exact/PCA/run2_good/algorithm_files/class_" + str(class_index) + "/fuzzy_QQplot/iterations_numpy/"
    itr = 600
    name = "X_matched_iteration_"+str(itr)
    X_class = load_variable(name_of_variable=name, path=path_)
    X = np.column_stack((X, X_class))
    color.extend([class_index] * X_class.shape[1])

# path2 = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/12_3blobs_3D_Exact/PCA/run2_good/algorithm_files/dim_reduction/PCA/figs/"
# color = load_variable(name_of_variable="X_low_dim_plotColors", path=path2)
# color = X[-1, :]

color_map = plt.cm.brg  #--> plt.cm.brg, plt.cm.hsv

plt.scatter(X[0, :], X[1, :], c=color, cmap=color_map, edgecolors='k')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.show()
plt.savefig(path_save + str(itr) + ".png")
plt.clf()
plt.close()

