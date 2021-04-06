from utils import *
import matplotlib.pyplot as plt

path_save = "C:/Users/benya/Desktop/"

X = np.empty((2, 0))
for class_index in range(10):
    path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/13_MNIST_Exact_supervised/MNIST_1000/algorithm_files/class_" + str(class_index) + "/fuzzy_QQplot/iterations_numpy/"
    itr = 600
    name = "X_matched_iteration_"+str(itr)
    X_class = load_variable(name_of_variable=name, path=path_)
    X = np.column_stack((X, X_class))

path2 = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/13_MNIST_Exact_supervised/MNIST_1000/algorithm_files/dim_reduction/PCA/figs/"
color = load_variable(name_of_variable="X_low_dim_plotColors", path=path2)
# color = X[-1, :]

color_map = plt.cm.tab10  #--> plt.cm.brg, plt.cm.hsv

# scatter_of_data_3(X=X, y=color, plot_name="plot", path_save_plot=path_save, color_map=color_map)
# X_outliersRemoved, color_meshgrid_outliersRemoved = remove_outliers(data_=X, color_meshgrid=color)

plt.scatter(X[0, :], X[1, :], c=color, cmap=color_map, edgecolors='k')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.show()
plt.savefig(path_save + str(itr) + ".png")
plt.clf()
plt.close()

