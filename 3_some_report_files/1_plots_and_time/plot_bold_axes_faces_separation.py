from utils import *
import matplotlib.pyplot as plt
# import matplotlib.colors

path_save = "C:/Users/benya/Desktop/"

# color_map =  matplotlib.colors.hsv_to_rgb(plt.cm.hsv) # plt.cm.bwr  #--> plt.cm.brg, plt.cm.hsv
# color_map = plt.cm.bwr

for i, plot_name in enumerate(["X_matched_initial", "X_matched_iteration_20", "X_matched_iteration_30", "X_matched_iteration_10"]):
    if i <= 2:
        class_index_of_plot = 0
    else:
        class_index_of_plot = 1
    path_ = "C:/Users/benya/Desktop/my_PhD/QQE/codes/4_results/17_face_glasses_transform_inputSpace/run2_good/algorithm_files/class_" + str(class_index_of_plot) + "/fuzzy_QQplot/iterations_umap/figs/"
    itr = "initial"
    X = load_variable(name_of_variable=plot_name+"_plotData", path=path_)
    y = load_variable(name_of_variable=plot_name+"_plotColors", path=path_)
    
    # plt.scatter(X[0, :], X[1, :], c=y, cmap=color_map, edgecolors='k')
    markers = ["v", "o"]
    colors = ["r", "b"]
    for class_index in range(2):
        sample_of_this_class = X[:, y == class_index]
        # c = class_index * np.ones((sample_of_this_class.shape[1],))
        plt.scatter(sample_of_this_class[0, :], sample_of_this_class[1, :], s=30, color=colors[class_index], alpha=1.0, marker=markers[class_index])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # plt.savefig(path_save + str(i) + ".png")
    # plt.clf()
    # plt.close()

    

