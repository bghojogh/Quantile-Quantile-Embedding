import pickle
import numpy as np
import csv
import os
from my_QQ_embedding import My_QQ_embedding
from my_Sammon_mapping import My_Sammon_mapping
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, samples_generator, make_circles, make_classification, make_s_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage.transform import resize
from PIL import Image
import math


def main():
    # ---- settings:
    dataset = "MNIST_1000_resnet" #--> one_blob, s_curve, three_blobs, three_clusters, ORL_glasses, ORL_glasses_balanced_separateClasses, MNIST_100, MNIST_500, MNIST_1000, MNIST_1000_resnet, MNIST_1000_siamese
    initialization_method = "" #--> PCA, Sammon_mapping, LLE, Isomap, TSNE, FDA
    experiment = 7
    dimensionaity_reduction_mode = False
    do_initialization_dimension_reduction_again = False
    generate_synthetic_datasets_again = False
    read_dataset_again = False
    plot_dataset_again = False
    transform_to_just_shape_of_reference = True
    generate_reference_sample_again = True
    swap_data_and_reference_sample = False
    notSupervisedButUseLabelsForPlot = False  #--> True: unsupervised (use labels for plot only if y is not None), False: supervised (or unsupervised when y=None)
    continue_from_previous_run = False
    which_iteration_to_load = 45
    match_points_again = True
    save_images_of_iterations = False
    save_umap_of_iterations = False
    embedding_dimensionality = 2
    n_samples_synthetic = 1000
    n_features_synthetic = 2
    n_neighbors = 10
    regularization_parameter = 10  #--> weight of QQ-plot (compared to the local distance preserving)
    learning_rate_fuzzy_QQplot = 0.01
    learning_rate_Sammon_mapping = 0.1
    max_iterations_Sammon_mapping = 11
    max_iterations_matching = 10
    max_iterations_transformation = 301
    colormap = plt.cm.tab10  #--> hsv, brg (good for S curve), rgb, jet, gist_ncar (good for one blob), tab10, Set1, rainbow, Spectral #--> https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html



    # ---- distribution transformation, synthetic:
    if dataset == "one_blob" or dataset == "s_curve" or dataset == "three_blobs" or dataset == "three_clusters":
        X, y = read_synthetic_dataset(dataset=dataset, generate_synthetic_datasets_again=generate_synthetic_datasets_again,
                                      plot_dataset_again=plot_dataset_again, n_samples=n_samples_synthetic,
                                      n_features=n_features_synthetic, colormap=colormap)
        X_images_to_plot = None
        image_height = None
        image_width = None
        plot_embedding_of_images = False
    elif dataset == "ORL_glasses":
        X, y, X_notNormalized = read_dataset(dataset=dataset, read_dataset_again=read_dataset_again)
        X_images_to_plot = X_notNormalized
        scale = 0.5
        image_height = int(112 * scale)
        image_width = int(92 * scale)
        plot_embedding_of_images = True
    elif dataset == "MNIST_100" or dataset == "MNIST_500" or dataset == "MNIST_1000":
        X, y, X_notNormalized = read_dataset(dataset=dataset, read_dataset_again=read_dataset_again)
        X_images_to_plot = X_notNormalized
        image_height = 28
        image_width = 28
        plot_embedding_of_images = True
    elif dataset == "ORL_glasses_balanced_separateClasses":
        X_no_glasses, X_with_glasses, _, _ = read_ORL_glasses_dataset(dataset, normalize=False)
        X = X_no_glasses
        y = None
        X_reference = X_with_glasses
        X_images_to_plot = None
        scale = 0.5
        image_height = int(112 * scale)
        image_width = int(92 * scale)
        plot_embedding_of_images = False
    elif dataset == "MNIST_1000_resnet":
        X = np.load("./datasets/MNIST_1000_resnet/embeddings_in_epoch_48.npy")
        X = X.T  #--> make it column-wise
        y = np.load("./datasets/MNIST_1000_resnet/subtypes_in_epoch_48.npy")
        print(X.shape)
        print(y.shape)
        fig, ax = plt.subplots()
        plt.scatter(X[0, :], X[1, :], c=y, cmap=colormap, edgecolors='k')
        plt.show()
        X_images_to_plot = None
        image_height = 28
        image_width = 28
        plot_embedding_of_images = False
    elif dataset == "MNIST_1000_siamese":
        take_1000_per_class_again = False
        if take_1000_per_class_again:
            X_all_triplets = np.load("./datasets/MNIST_1000_siamese/embeddings_in_epoch_30.npy")
            X_all_triplets = X_all_triplets.T  #--> make it column-wise
            y_all_triplets = np.load("./datasets/MNIST_1000_siamese/labels_in_epoch_30.npy")
            X = np.empty((2, 0))
            y = []
            for class_index in range(10):
                print("class: " + str(class_index))
                X_all_triplets_thisClass = X_all_triplets[:, y_all_triplets==class_index]
                y_all_triplets_thisClass = y_all_triplets[y_all_triplets==class_index]
                counter_ = -1
                X_thisClass = np.zeros((X_all_triplets.shape[0], 1000))
                y_thisClass = np.zeros((1000,))
                for sample_index in range(X_all_triplets_thisClass.shape[1]):
                    repeated = False
                    for sample_index_2 in range(0, sample_index):
                        if np.linalg.norm(X_all_triplets_thisClass[:, sample_index] - X_all_triplets_thisClass[:, sample_index_2]) == 0: #--> is a reapted sample
                            repeated = True
                    if not repeated:
                        counter_ += 1
                        if counter_ >= 1000:
                            break
                        X_thisClass[:, counter_] = X_all_triplets_thisClass[:, sample_index]
                        y_thisClass[counter_] = y_all_triplets_thisClass[sample_index]
                # X_thisClass = X_all_triplets_thisClass[:, :1000]
                X = np.column_stack((X, X_thisClass))
                y.extend(y_all_triplets_thisClass[:1000])
            y = np.asarray(y)
            save_variable(variable=X, name_of_variable="X", path_to_save="./datasets/MNIST_1000_siamese/")
            save_variable(variable=y, name_of_variable="y", path_to_save="./datasets/MNIST_1000_siamese/")
        else:
            X = load_variable(name_of_variable="X", path="./datasets/MNIST_1000_siamese/")
            y = load_variable(name_of_variable="y", path="./datasets/MNIST_1000_siamese/")
        fig, ax = plt.subplots()
        plt.scatter(X[0, :], X[1, :], c=y, cmap=colormap, edgecolors='k')
        plt.show()
        X_images_to_plot = None
        image_height = 28
        image_width = 28
        plot_embedding_of_images = False


    if experiment == 1:
        reference_distribution = ["uniform", "uniform", "uniform"]
        reference_distribution_parameters = [[0.5, 1.5], [1, 2], [-2, -1]]
    elif experiment == 2:
        reference_distribution = "uniform"
        reference_distribution_parameters = [0.5, 1.5]
        y = None
    elif experiment == 3:
        reference_distribution = "uniform"
        reference_distribution_parameters = [0.5, 1.5]
    elif experiment == 4:
        reference_distribution = ["uniform", "uniform"]
        reference_distribution_parameters = [[1, 2], [-2, -1]]
    elif experiment == 5:
        reference_distribution = ["blob", "blob", "blob"]
        reference_distribution_parameters = [[]] * 3
        reference_distribution_parameters[0] = [[3, 3], 0.5]  # --> [centers, cluster_std]
        reference_distribution_parameters[1] = [[-3, -2.5], 1.5]  # --> [centers, cluster_std]
        reference_distribution_parameters[2] = [[5, -2.5], 1]  # --> [centers, cluster_std]
    elif experiment == 6:
        reference_distribution = ["uniform", "uniform", "uniform"]
        reference_distribution_parameters = [[1, 2], [-0.5, 0.5], [-2, -1]]
    elif experiment == 7:
        reference_distribution = ["swiss_roll", "swiss_roll", "circle", "circle", "s_curve", "s_curve",
                                  "uniform_manual", "uniform_manual", "blob", "blob"]
        reference_distribution_parameters = [[]] * 10
        reference_distribution_parameters[0] = [[-5, 0], 1.5]  # --> [mean, radius]
        reference_distribution_parameters[1] = [[-3.5, 3.5], 1.5]  # --> [mean, radius]
        reference_distribution_parameters[2] = [[0, 5], 1]  # --> [mean, radius]
        reference_distribution_parameters[3] = [[3.5, 3.5], 1]  # --> [mean, radius]
        reference_distribution_parameters[4] = [[5, 0], 0.65]  # --> [mean, radius]
        reference_distribution_parameters[5] = [[3.5, -3.5], 0.65]  # --> [mean, radius]
        reference_distribution_parameters[6] = [[-1, 1], [-6, -4]]  # --> [[min in 1st dimension, max in 1st dimension], [min in 2nd dimension, max in 2nd dimension]]
        reference_distribution_parameters[7] = [[-4.5, -2.5], [-4.5, -2.5]]  # --> [[min in 1st dimension, max in 1st dimension], [min in 2nd dimension, max in 2nd dimension]]
        reference_distribution_parameters[8] = [[-1.8, 0], 0.5]  # --> [center, cluster_std]
        reference_distribution_parameters[9] = [[1.8, 0], 0.5]  # --> [center, cluster_std]
    elif experiment == 8:
        reference_distribution = "thick_circle"
        reference_distribution_parameters = [0.5, 1.5, [1, 1]]  # --> [inner radius, outer radius, mean]
    elif experiment == 9:
        reference_distribution = "triangle"
        reference_distribution_parameters = [[1, 1], 1]  # --> [mean, scale]
    elif experiment == 10:
        reference_distribution = "thick_circle"  #--> filled circle
        reference_distribution_parameters = [0, 1.5, [1, 1]]  # --> [inner radius, outer radius, mean]
    elif experiment == 11:
        reference_distribution = "two_blobs"
        reference_distribution_parameters = [[]] * 2
        reference_distribution_parameters[0] = [[-1, 0], [11, 0]]  # --> centers
        reference_distribution_parameters[1] = [1.5, 3]  # --> cluster_std
    elif experiment == 12:
        reference_distribution = "load_reference_sample"
        reference_distribution_parameters = X_reference
    elif experiment == 13:
        n_dimensions = X.shape[0]
        reference_distribution = ["Gaussian", "Gaussian"]
        reference_distribution_parameters = [[]] * 2
        reference_distribution_parameters[0] = [-5 * np.ones((n_dimensions,)), 1 * np.eye(n_dimensions)]  # --> [mean, cov]
        reference_distribution_parameters[1] = [5 * np.ones((n_dimensions,)), 1 * np.eye(n_dimensions)]  # --> [mean, cov]
    elif experiment == 14:
        y = None
        # CDF:
        CDF_x_plot = [-5, -2, -1, 1, 2, 5]
        CDF_y_plot = [0, 0.05, 0.45, 0.55, 0.95, 1]
        plt.plot(CDF_x_plot, CDF_y_plot, "-", color="b", linewidth=5)
        plt.xticks(np.arange(min(CDF_x_plot), max(CDF_x_plot) + 1, 1), fontsize=15)
        plt.yticks(np.arange(min(CDF_y_plot), max(CDF_y_plot)+0.1, 0.1), fontsize=15)
        plt.xlabel("x", fontsize=15)
        plt.ylabel("CDF(x)", fontsize=15)
        plt.grid()
        plt.show()
        # inverse CDF:
        CDF_x_plot = [-5, -2, -1, 1, 2, 5]
        CDF_y_plot = [0, 0.05, 0.45, 0.55, 0.95, 1]
        plt.plot(CDF_y_plot, CDF_x_plot, "-", color="b", linewidth=5)
        plt.yticks(np.arange(min(CDF_x_plot), max(CDF_x_plot) + 1, 1), fontsize=15)
        plt.xticks(np.arange(min(CDF_y_plot), max(CDF_y_plot) + 0.1, 0.1), fontsize=15)
        plt.xlabel("x", fontsize=15)
        plt.ylabel("Inverse CDF(x)", fontsize=15)
        plt.grid()
        plt.show()
        # set the marginal CDF functions:
        reference_distribution = "CDF_onlyFirstDimension"
        reference_distribution_parameters = [[]] * 6  #--> 5 lines in inverse CDF + 1 for the x breaks in inverse CDF
        reference_distribution_parameters[0] = [60, -5]  #--> [slope, intercept] of line in inverse CDF
        reference_distribution_parameters[1] = [2.5, -2.125]  # --> [slope, intercept] of line in inverse CDF
        reference_distribution_parameters[2] = [20, -10]  # --> [slope, intercept] of line in inverse CDF
        reference_distribution_parameters[3] = [2.5, -0.375]  # --> [slope, intercept] of line in inverse CDF
        reference_distribution_parameters[4] = [60, -55]  # --> [slope, intercept] of line in inverse CDF
        reference_distribution_parameters[5] = [0.05, 0.45, 0.55, 0.95, 1]  #--> the x breaks in inverse CDF
    elif experiment == 15:
        reference_distribution = "diamond"
        reference_distribution_parameters = [[1, 1], 1]  # --> [mean, scale]
    elif experiment == 16:
        reference_distribution = "thick_square"
        reference_distribution_parameters = [0.5, 1.5, [1, 1]]  # --> [inner radius, outer radius, mean]

    my_QQ_embedding = My_QQ_embedding(learning_rate_fuzzy_QQplot=learning_rate_fuzzy_QQplot, learning_rate_Sammon_mapping=learning_rate_Sammon_mapping,
                                      regularization_parameter=regularization_parameter, n_neighbors=n_neighbors,
                                      transform_to_just_shape_of_reference=transform_to_just_shape_of_reference,
                                      max_iterations_Sammon_mapping=max_iterations_Sammon_mapping, max_iterations_matching=max_iterations_matching,
                                      max_iterations_transformation=max_iterations_transformation, colormap=colormap,
                                      X_images_to_plot=X_images_to_plot, image_height=image_height, image_width=image_width,
                                      plot_embedding_of_images=plot_embedding_of_images, notSupervisedButUseLabelsForPlot=notSupervisedButUseLabelsForPlot,
                                      continue_from_previous_run=continue_from_previous_run, which_iteration_to_load=which_iteration_to_load,
                                      match_points_again=match_points_again, save_images_of_iterations=save_images_of_iterations, save_umap_of_iterations=save_umap_of_iterations)


    if dimensionaity_reduction_mode:
        X_embedded = my_QQ_embedding.dimensionality_reduction(X=X, y=y, embedding_dimensionality=embedding_dimensionality,
                                                              reference_distribution=reference_distribution, reference_distribution_parameters=reference_distribution_parameters,
                                                              CDF=None, generate_reference_sample_again=generate_reference_sample_again,
                                                              initialization_method=initialization_method, do_initialization_dimension_reduction_again=do_initialization_dimension_reduction_again)
    else:
        X_embedded = my_QQ_embedding.distribution_transformation(X=X, y=y,
                                                              reference_distribution=reference_distribution, reference_distribution_parameters=reference_distribution_parameters,
                                                              CDF=None, generate_reference_sample_again=generate_reference_sample_again,
                                                              swap_data_and_reference_sample=swap_data_and_reference_sample)


def read_synthetic_dataset(dataset, generate_synthetic_datasets_again, plot_dataset_again, n_samples, n_features, colormap):
    # https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
    # settings:
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
    rng = np.random.RandomState(42)
    # dataset:
    path_dataset = './datasets/' + dataset + "/dimension_" + str(n_features) + "/"
    if generate_synthetic_datasets_again:
        if dataset == "one_blob":
            supervised = False
            if n_features == 2:
                X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[1, 1]], cluster_std=[0.5])
            else:
                X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[1, 1, 1]], cluster_std=[1])
        elif dataset == "s_curve":
            supervised = False
            X_3d, y = make_s_curve(n_samples=n_samples, random_state=0)
            if n_features == 2:
                X = np.column_stack((X_3d[:, 0], X_3d[:, 2]))
            else:
                X = X_3d
        elif dataset == "three_blobs":
            supervised = True
            if n_features == 2:
                # X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[1, 1], [-3, -2.5], [2.5, -2.5]], cluster_std=[0.5, 1.5, 1])
                X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[-1, -1], [-3, -2.5], [-1, -2.5]], cluster_std=[0.5, 1.5, 1])
            else:
                # X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[1, 1, 1], [-3, -2.5, 2], [5, -2.5, 4]], cluster_std=[0.5, 1.5, 1])
                # X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[-1, -1, 1], [-3, -2.5, 2], [-1, -2.5, 4]], cluster_std=[0.5, 1.5, 1])
                # X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[-1, -1, 1], [-3, -2.5, 1], [-1, -2.5, 1]], cluster_std=[0.5, 1.5, 1])
                X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[[-1, -1, 5], [-3, -2.5, 2], [-1, -2.5, -5]], cluster_std=[0.5, 1.5, 1])
        elif dataset == "three_clusters":
            # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
            # https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html
            supervised = True
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3)
        elif dataset == "uniform_grid":
            x = np.linspace(0, 1, int(np.sqrt(300)))
            xx, yy = np.meshgrid(x, x)
            X = np.hstack([
                xx.ravel().reshape(-1, 1),
                yy.ravel().reshape(-1, 1),
            ])
            y = xx.ravel()
        X = StandardScaler().fit_transform(X)
        save_variable(variable=X, name_of_variable="X", path_to_save=path_dataset)
        save_variable(variable=y, name_of_variable="y", path_to_save=path_dataset)
    else:
        X = load_variable(name_of_variable="X", path=path_dataset)
        y = load_variable(name_of_variable="y", path=path_dataset)
        if dataset == "one_blob":
            supervised = False
        elif dataset == "s_curve":
            supervised = False
        elif dataset == "three_blobs":
            supervised = True
        elif dataset == "three_clusters":
            supervised = True
    if plot_dataset_again:
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # just plot the dataset first
        # cm_bright = plt.cm.RdBu
        n_classes = len(np.unique(y))
        if n_classes == 2:
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        else:
            # cm_bright = plt.cm.brg
            # cm_bright = plt.cm.hsv
            cm_bright = colormap
        # cm_bright = ListedColormap(['#FF0000', '#0000FF', '#FF00FF'])
        if n_features == 2:
            ax = plt.subplot()
        elif n_features == 3:
            ax = plt.subplot(projection='3d')
        # Plot the training points
        if n_features == 2:
            if supervised:
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
            else:
                ax.scatter(X[:, 0], X[:, 1], c=X[:, -1], cmap=cm_bright, edgecolors='k')
        else:
            if supervised:
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cm_bright, edgecolors='k')
            else:
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, -1], cmap=cm_bright, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # ax.set_xticks(())
        # ax.set_yticks(())
        plt.show()
    # make X column-wise:
    X = X.T
    return X, y

def read_dataset(dataset, read_dataset_again):
    if dataset == "ORL_faces":
        # scale = 0.4
        # scale = 0.6
        scale = 1
        path_dataset = "./datasets/ORL_faces/"
        image_height, image_width = 112, 92
        n_classes = 20
        if read_dataset_again:
            X, y = read_ORL_dataset(path_dataset=path_dataset, image_height=image_height, image_width=image_width,
                                    n_classes=n_classes, do_resize=False, scale=scale)
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="X", arr=X)
            save_numpy_array(path_to_save="./datasets/ORL/", arr_name="y", arr=y)
        else:
            X = np.load("./datasets/ORL/X.npy")
            y = np.load("./datasets/ORL/y.npy")
    elif dataset == "ORL_glasses":
        path_dataset = "./datasets/ORL_glasses/"
        n_samples = 400
        scale = 0.5
        image_height = int(112 * scale)
        image_width = int(92 * scale)
        data = np.zeros((image_height * image_width, n_samples))
        labels = np.zeros((1, n_samples))
        image_index = -1
        for class_index in range(2):
            for filename in os.listdir(path_dataset + "class" + str(class_index + 1) + "/"):
                image_index = image_index + 1
                if image_index >= n_samples:
                    break
                img = load_image(address_image=path_dataset + "class" + str(class_index + 1) + "/" + filename,
                                 image_height=image_height, image_width=image_width, do_resize=False, scale=scale)
                data[:, image_index] = img.ravel()
                labels[:, image_index] = class_index
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- normalize (standardation):
        X_notNormalized = data
        # data = data / 255
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T
        X = data
        y = labels
    elif dataset == 'MNIST_100' or dataset == 'MNIST_500' or dataset == 'MNIST_1000':
        subset_of_MNIST = True
        pick_subset_of_MNIST_again = True
        if dataset == "MNIST_100":
            MNIST_subset_cardinality_training = 100
        elif dataset == "MNIST_500":
            MNIST_subset_cardinality_training = 500
        elif dataset == "MNIST_1000":
            MNIST_subset_cardinality_training = 1000
        MNIST_subset_cardinality_testing = 10
        path_dataset_save = './datasets/MNIST/'
        file = open(path_dataset_save+'X_train.pckl','rb')
        X_train = pickle.load(file); file.close()
        file = open(path_dataset_save+'y_train.pckl','rb')
        y_train = pickle.load(file); file.close()
        file = open(path_dataset_save+'X_test.pckl','rb')
        X_test = pickle.load(file); file.close()
        file = open(path_dataset_save+'y_test.pckl','rb')
        y_test = pickle.load(file); file.close()
        if subset_of_MNIST:
            if pick_subset_of_MNIST_again and read_dataset_again:
                dimension_of_data = 28 * 28
                X_train_picked = np.empty((0, dimension_of_data))
                y_train_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_train[y_train == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_training, :]
                    X_train_picked = np.vstack((X_train_picked, X_class_picked))
                    y_class = y_train[y_train == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_training].reshape((-1, 1))
                    y_train_picked = np.vstack((y_train_picked, y_class_picked))
                y_train_picked = y_train_picked.ravel()
                X_test_picked = np.empty((0, dimension_of_data))
                y_test_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_test[y_test == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_testing, :]
                    X_test_picked = np.vstack((X_test_picked, X_class_picked))
                    y_class = y_test[y_test == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_testing].reshape((-1, 1))
                    y_test_picked = np.vstack((y_test_picked, y_class_picked))
                y_test_picked = y_test_picked.ravel()
                # X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                # X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                # y_train_picked = y_train[0:MNIST_subset_cardinality_training]
                # y_test_picked = y_test[0:MNIST_subset_cardinality_testing]
                save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset_save+dataset+"/")
                save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset_save+dataset+"/")
                save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset_save+dataset+"/")
                save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset_save+dataset+"/")
            else:
                file = open(path_dataset_save+dataset+"/"+'X_train_picked.pckl','rb')
                X_train_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+dataset+"/"+'X_test_picked.pckl','rb')
                X_test_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+dataset+"/"+'y_train_picked.pckl','rb')
                y_train_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+dataset+"/"+'y_test_picked.pckl','rb')
                y_test_picked = pickle.load(file); file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            y_train = y_train_picked
            y_test = y_test_picked
        data = X_train.T / 255
        data_test = X_test.T / 255
        labels = y_train.reshape((1, -1))
        n_samples = data.shape[1]
        image_height = 28
        image_width = 28
        # ---- normalize (standardation):
        X_notNormalized = data
        # data = data / 255
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)  #--> comment it for LLE on MNIST
        data = (scaler.transform(data.T)).T   #--> comment it for LLE on MNIST
        X = data
        y = labels
    y = y.ravel()
    # print(np.max(X))
    # input("hi")
    return X, y, X_notNormalized

def read_ORL_glasses_dataset(dataset, normalize=False):
    if dataset == "ORL_glasses_balanced_separateClasses":
        path_dataset = "./datasets/ORL_glasses_balanced/"
        n_samples_per_class = 119
        scale = 0.5
        image_height = int(112 * scale)
        image_width = int(92 * scale)
        for class_index in range(2):
            data = np.zeros((image_height * image_width, n_samples_per_class))
            image_index = -1
            for filename in os.listdir(path_dataset + "class" + str(class_index + 1) + "/"):
                image_index = image_index + 1
                if image_index >= n_samples_per_class:
                    break
                img = load_image(address_image=path_dataset + "class" + str(class_index + 1) + "/" + filename,
                                 image_height=image_height, image_width=image_width, do_resize=False, scale=scale)
                data[:, image_index] = img.ravel()
            if class_index == 0:
                X_no_glasses = data.copy()
            else:
                X_with_glasses = data.copy()
        # ---- cast dataset from string to float:
        X_no_glasses = X_no_glasses.astype(np.float)
        X_with_glasses = X_with_glasses.astype(np.float)
        # ---- normalize (standardation):
        X_no_glasses_notNormalized = X_no_glasses
        X_with_glasses_notNormalized = X_with_glasses
        # data = data / 255
        if normalize:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_no_glasses.T)
            X_no_glasses = (scaler.transform(X_no_glasses.T)).T
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_with_glasses.T)
            X_with_glasses = (scaler.transform(X_with_glasses.T)).T
    return X_no_glasses, X_with_glasses, X_no_glasses_notNormalized, X_with_glasses_notNormalized

def read_ORL_dataset(path_dataset, image_height, image_width, n_classes=None, do_resize=False, scale=1):
    if n_classes is None:
        n_samples = 400
    else:
        n_samples = n_classes * 10
    if do_resize:
        data = np.zeros((int(image_height * scale) * int(image_width * scale), n_samples))
    else:
        data = np.zeros((image_height*image_width, n_samples))
    labels = np.zeros((1, n_samples))
    for image_index in range(n_samples):
        img = load_image(address_image=path_dataset+str(image_index+1)+".jpg",
                        image_height=image_height, image_width=image_width, do_resize=do_resize, scale=scale)
        data[:, image_index] = img.ravel()
        labels[:, image_index] = math.floor(image_index / 10) + 1
    # ---- cast dataset from string to float:
    data = data.astype(np.float)
    # ---- change range of images from [0,255] to [0,1]:
    data = data / 255
    data_notNormalized = data
    # ---- normalize (standardation):
    scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
    data = (scaler.transform(data.T)).T
    # ---- show one of the images:
    print("dimensionality: " + str(data.shape[0]))
    # if False:
    #     if resize:
    #         an_image = data[:, 0].reshape((int(image_height * scale), int(image_width * scale)))
    #     else:
    #         an_image = data[:, 0].reshape((image_height, image_width))
    #     plt.imshow(an_image, cmap='gray')
    #     plt.colorbar()
    #     plt.show()
    return data, labels.ravel()

def load_image(address_image, image_height, image_width, do_resize=False, scale=1):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    if do_resize:
        size = int(image_height * scale), int(image_width * scale)
        # img.thumbnail(size, Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = resize(img_arr, (int(img_arr.shape[0]*scale), int(img_arr.shape[1]*scale)), order=5, preserve_range=True, mode="constant")
    return img_arr

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def save_numpy_array(path_to_save, arr_name, arr):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    np.save(path_to_save+arr_name+".npy", arr)

if __name__ == '__main__':
    main()