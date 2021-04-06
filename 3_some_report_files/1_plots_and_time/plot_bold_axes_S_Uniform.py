from utils import *
import matplotlib.pyplot as plt

path_save = "C:/Users/benya/Desktop/"

# path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/5_S_2D/run3_good/algorithm_files/fuzzy_QQplot/figs/"
# itr = "X_initial"
# name = "X_matched_initial_plotData"

# path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/5_S_2D/run3_good/algorithm_files/matched_data/"
# itr = "Y_initial"
# name = "Y_matched"

# path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/5_S_2D/run3_good/algorithm_files/fuzzy_QQplot/iterations_numpy/"
# itr = 180
# name = "X_matched_iteration_"+str(itr)

# path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/6_S_2D_swap/algorithm_files/fuzzy_QQplot/iterations_numpy/"
# itr = 90
# name = "X_matched_iteration_"+str(itr)

# path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/7_S_2D_Exact/algorithm_files/fuzzy_QQplot/iterations_numpy/"
# itr = 215
# name = "X_matched_iteration_"+str(itr)

path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/8_S_2D_Exact_swap/algorithm_files/fuzzy_QQplot/iterations_numpy/"
itr = 605
name = "X_matched_iteration_"+str(itr)


color_map = plt.cm.hsv  #--> plt.cm.brg, plt.cm.hsv

X = load_variable(name_of_variable=name, path=path_)

path2 = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/5_S_2D/run3_good/algorithm_files/fuzzy_QQplot/figs/"
color = load_variable(name_of_variable="Y_matched_plotColors", path=path2)
# color = X[-1, :]

X_outliersRemoved, color_meshgrid_outliersRemoved = remove_outliers(data_=X, color_meshgrid=color)

plt.scatter(X_outliersRemoved[0, :], X_outliersRemoved[1, :], c=color_meshgrid_outliersRemoved, cmap=color_map, edgecolors='k')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.show()
plt.savefig(path_save + str(itr) + ".png")
plt.clf()
plt.close()

