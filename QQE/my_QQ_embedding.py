from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from my_Sammon_mapping import My_Sammon_mapping
from matplotlib import offsetbox
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time
from sklearn.datasets import make_blobs, make_s_curve, make_circles, make_swiss_roll
from PIL import Image
import umap
import random


import warnings
warnings.filterwarnings("ignore")

class My_QQ_embedding:

    def __init__(self, learning_rate_fuzzy_QQplot=0.1, learning_rate_Sammon_mapping=0.1, regularization_parameter=None, n_neighbors=None,
                 transform_to_just_shape_of_reference=True, max_iterations_Sammon_mapping=100, max_iterations_matching=100,
                 max_iterations_transformation=100, colormap=plt.cm.brg, X_images_to_plot=None, image_height=None, image_width=None,
                 plot_embedding_of_images=False, notSupervisedButUseLabelsForPlot=False, continue_from_previous_run=False, which_iteration_to_load=0,
                 match_points_again=True, save_images_of_iterations=False, save_umap_of_iterations=False):
        self.X = None
        self.n_dimensions = None
        self.n_samples = None
        self.n_samples_reference = None
        self.n_dimensions_reference = None
        if n_neighbors is None:
            self.n_neighbors = self.n_samples - 1
        else:
            self.n_neighbors = n_neighbors
        if regularization_parameter is None:
            self.regularization_parameter = 1.0
        else:
            self.regularization_parameter = regularization_parameter
        self.learning_rate_fuzzy_QQplot = learning_rate_fuzzy_QQplot
        self.learning_rate_Sammon_mapping = learning_rate_Sammon_mapping
        self.dataset_can_be_plotted = None
        self.reference_sample_can_be_plotted = None
        self.transform_to_just_shape_of_reference = transform_to_just_shape_of_reference
        self.max_iterations_Sammon_mapping = max_iterations_Sammon_mapping
        self.max_iterations_matching = max_iterations_matching
        self.max_iterations_transformation = max_iterations_transformation
        self.color_meshgrid = None
        self.colormap = colormap
        self.supervised_mode = None
        self.X_multiclass_to_plot = None
        self.which_class_workingOn_now = None
        self.labels_of_all_dataset = None
        self.indices_of_points_in_classes = None
        self.reorder_reference_sample_in_matching = True
        self.X_images_to_plot = X_images_to_plot
        self.y = None
        self.image_height = image_height
        self.image_width = image_width
        self.plot_embedding_of_images = plot_embedding_of_images
        self.notSupervisedButUseLabelsForPlot = notSupervisedButUseLabelsForPlot
        self.continue_from_previous_run = continue_from_previous_run
        self.which_iteration_to_load = which_iteration_to_load
        self.match_points_again = match_points_again
        self.save_images_of_iterations = save_images_of_iterations
        self.save_umap_of_iterations = save_umap_of_iterations

    def distribution_transformation(self, X, y=None,
                                 reference_distribution="uniform", reference_distribution_parameters=None, CDF=None,
                                 generate_reference_sample_again=True, swap_data_and_reference_sample=False):
        # X: samples are put column-wise in matrix
        self.y = y
        if (y is not None) and (not self.notSupervisedButUseLabelsForPlot):
            self.supervised_mode = True
        else:
            self.supervised_mode = False
        if (not self.supervised_mode) and (not self.notSupervisedButUseLabelsForPlot):
            self.color_meshgrid = X[-1, :]
        else:
            assert y is not None
            self.color_meshgrid = y
            self.labels_of_all_dataset = y
        self.n_dimensions_reference = X.shape[0]
        if X.shape[0] == 2 or X.shape[0] == 3:
            self.dataset_can_be_plotted = True
            self.reference_sample_can_be_plotted = True
        else:
            self.dataset_can_be_plotted = False
            self.reference_sample_can_be_plotted = False
        if not self.supervised_mode:
            self.X = X
            self.n_dimensions = self.X.shape[0]
            self.n_samples = self.X.shape[1]
            self.make_reference_sample(distribution=reference_distribution, CDF=CDF, reference_distribution_parameters=reference_distribution_parameters, n_samples_reference=None, n_dimensions_reference=None,
                                        plot_reference_distribution=False, generate_reference_sample_again=generate_reference_sample_again, path_save_base="./algorithm_files/")
            if swap_data_and_reference_sample:
                temp_ = self.reference_sample.copy()
                self.reference_sample = self.X.copy()
                self.X = temp_.copy()
                self.reorder_reference_sample_in_matching = False
            X_embedded = self.distribution_transformation_fuzzy_QQplot(max_iterations_matching=self.max_iterations_matching, max_iterations_transformation=self.max_iterations_transformation,
                                                                        save_each_how_many_epochs=5, continue_from_previous_run=self.continue_from_previous_run, which_iteration_to_load=self.which_iteration_to_load, path_save_base="./algorithm_files/", match_points_again=self.match_points_again)
        else:
            X_separated_classes, self.indices_of_points_in_classes = self.separate_samples_of_classes_2(X=X, y=y)
            n_classes = len(X_separated_classes)
            self.X_multiclass_to_plot = X.copy()
            self.labels_of_all_dataset = y
            for class_index in range(n_classes):
                self.which_class_workingOn_now = class_index
                print("================ Working on class #" + str(class_index) + ":")
                self.X = X_separated_classes[class_index].copy()
                self.n_dimensions = self.X.shape[0]
                self.n_samples = self.X.shape[1]
                self.color_meshgrid = np.ones((X_separated_classes[class_index].shape[1],)) * class_index
                self.make_reference_sample(distribution=reference_distribution[class_index], CDF=CDF, reference_distribution_parameters=reference_distribution_parameters[class_index], n_samples_reference=None, n_dimensions_reference=None,
                                        plot_reference_distribution=False, generate_reference_sample_again=generate_reference_sample_again, path_save_base="./algorithm_files/class_"+str(class_index)+"/")
                X_embedded = self.distribution_transformation_fuzzy_QQplot(max_iterations_matching=self.max_iterations_matching, max_iterations_transformation=self.max_iterations_transformation,
                                                                            save_each_how_many_epochs=5, continue_from_previous_run=self.continue_from_previous_run, which_iteration_to_load=self.which_iteration_to_load, path_save_base="./algorithm_files/class_"+str(class_index)+"/", match_points_again=self.match_points_again)
                self.X_multiclass_to_plot[:, self.indices_of_points_in_classes[self.which_class_workingOn_now]] = X_embedded
            X_embedded = self.X_multiclass_to_plot  #--> total embedded data (of all classes)
        return X_embedded

    def dimensionality_reduction(self, X, y=None, embedding_dimensionality=2,
                                 reference_distribution="uniform", reference_distribution_parameters=None, CDF=None,
                                 generate_reference_sample_again=True, initialization_method="PCA", do_initialization_dimension_reduction_again=False):
        # X: samples are put column-wise in matrix
        self.y = y
        if (y is not None) and (not self.notSupervisedButUseLabelsForPlot):
            self.supervised_mode = True
        else:
            self.supervised_mode = False
        if (not self.supervised_mode) and (not self.notSupervisedButUseLabelsForPlot):
            self.color_meshgrid = X[-1, :]
        else:
            assert y is not None
            self.color_meshgrid = y
            self.labels_of_all_dataset = y
        self.n_dimensions_reference = embedding_dimensionality
        if X.shape[0] == 2 or X.shape[0] == 3:
            input_dataset_can_be_plotted = True
        else:
            input_dataset_can_be_plotted = False
        if embedding_dimensionality == 2 or embedding_dimensionality == 3:
            embedded_data_can_be_plotted = True
            self.dataset_can_be_plotted = True
            self.reference_sample_can_be_plotted = True
        else:
            embedded_data_can_be_plotted = False
            self.dataset_can_be_plotted = False
            self.reference_sample_can_be_plotted = False
        if do_initialization_dimension_reduction_again:
            if initialization_method == "Sammon_mapping":
                my_Sammon_mapping = My_Sammon_mapping(X=X, embedding_dimensionality=embedding_dimensionality, n_neighbors=self.n_neighbors, dataset_can_be_plotted=input_dataset_can_be_plotted,
                                                      embedded_data_can_be_plotted=embedded_data_can_be_plotted, max_iterations=self.max_iterations_Sammon_mapping,
                                                      learning_rate=self.learning_rate_Sammon_mapping, init_type="PCA", color_meshgrid=self.color_meshgrid, colormap=self.colormap)
                X_low_dim = my_Sammon_mapping.fit_transform(continue_from_previous_run=False, which_iteration_to_load=0)
            elif initialization_method == "PCA":
                pca = PCA(n_components=embedding_dimensionality)
                X_low_dim = (pca.fit_transform(X.T)).T
            elif initialization_method == "LLE":
                # lle = LLE(n_components=embedding_dimensionality, n_neighbors=self.n_neighbors)  #--> use this for LLE on other datasets
                lle = LLE(n_components=embedding_dimensionality)  #--> use this for LLE on MNIST
                X_low_dim = (lle.fit_transform(X.T)).T
                X_low_dim *= 150 #--> scale it to become larger
            elif initialization_method == "Isomap":
                isomap = Isomap(n_components=embedding_dimensionality, n_neighbors=self.n_neighbors)
                X_low_dim = (isomap.fit_transform(X.T)).T
            elif initialization_method == "TSNE":
                tsne = TSNE(n_components=embedding_dimensionality)
                X_low_dim = (tsne.fit_transform(X.T)).T
            elif initialization_method == "FDA":
                lda = LDA(n_components=embedding_dimensionality)
                X_low_dim = (lda.fit_transform(X.T, y)).T
            path_save_base = "./algorithm_files/dim_reduction/"+str(initialization_method)+"/"
            self.save_scatter_of_data(data_=X_low_dim, data_name="X_low_dim", path_save_numpy=path_save_base, path_save_plot=path_save_base, color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.reference_sample_can_be_plotted)
        else:
            path_save_base = "./algorithm_files/dim_reduction/"+str(initialization_method)+"/"
            X_low_dim = self.load_variable(name_of_variable="X_low_dim", path=path_save_base)
        # plot the legend of classes:
        if y is not None and embedding_dimensionality <= 3:
            plt.scatter(X_low_dim[0, :], X_low_dim[1, :], c=self.color_meshgrid, cmap=self.colormap, edgecolors='k')
            n_classes = len(np.unique(y))
            classes = [str(i) for i in range(n_classes)]
            cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
            cbar.set_ticks(np.arange(n_classes))
            cbar.set_ticklabels(classes)
            # plt.show()
            plt.savefig(path_save_base + "X_low_dim_LEGEND" + ".png")
            plt.clf()
            # plt.close()
        if not self.supervised_mode:
            self.X = X_low_dim
            self.n_dimensions = self.X.shape[0]
            self.n_samples = self.X.shape[1]
            self.make_reference_sample(distribution=reference_distribution, CDF=CDF, reference_distribution_parameters=reference_distribution_parameters, n_samples_reference=None, n_dimensions_reference=None,
                                        plot_reference_distribution=False, generate_reference_sample_again=generate_reference_sample_again, path_save_base="./algorithm_files/")
            X_embedded = self.distribution_transformation_fuzzy_QQplot(max_iterations_matching=self.max_iterations_matching, max_iterations_transformation=self.max_iterations_transformation,
                                                                        save_each_how_many_epochs=5, continue_from_previous_run=self.continue_from_previous_run, which_iteration_to_load=self.which_iteration_to_load, path_save_base="./algorithm_files/", match_points_again=self.match_points_again)
        else:
            X_low_dim_separated_classes, self.indices_of_points_in_classes = self.separate_samples_of_classes_2(X=X_low_dim, y=y)
            n_classes = len(X_low_dim_separated_classes)
            self.X_multiclass_to_plot = X_low_dim.copy()
            self.labels_of_all_dataset = y
            for class_index in range(n_classes):
                self.which_class_workingOn_now = class_index
                print("================ Working on class #" + str(class_index) + ":")
                self.X = X_low_dim_separated_classes[class_index].copy()
                self.n_dimensions = self.X.shape[0]
                self.n_samples = self.X.shape[1]
                self.color_meshgrid = np.ones((X_low_dim_separated_classes[class_index].shape[1],)) * class_index
                self.make_reference_sample(distribution=reference_distribution[class_index], CDF=CDF, reference_distribution_parameters=reference_distribution_parameters[class_index], n_samples_reference=None, n_dimensions_reference=None,
                                        plot_reference_distribution=False, generate_reference_sample_again=generate_reference_sample_again, path_save_base="./algorithm_files/class_"+str(class_index)+"/")
                X_embedded = self.distribution_transformation_fuzzy_QQplot(max_iterations_matching=self.max_iterations_matching, max_iterations_transformation=self.max_iterations_transformation,
                                                                            save_each_how_many_epochs=5, continue_from_previous_run=self.continue_from_previous_run, which_iteration_to_load=self.which_iteration_to_load, path_save_base="./algorithm_files/class_"+str(class_index)+"/", match_points_again=self.match_points_again)
                self.X_multiclass_to_plot[:, self.indices_of_points_in_classes[self.which_class_workingOn_now]] = X_embedded
            X_embedded = self.X_multiclass_to_plot  #--> total embedded data (of all classes)
        return X_embedded

    def make_reference_sample(self, distribution, CDF=None, reference_distribution_parameters=None, n_samples_reference=None, n_dimensions_reference=None,
                              plot_reference_distribution=False, generate_reference_sample_again=True, path_save_base="./"):
        # self.reference_sample: will be column-wise
        if generate_reference_sample_again:
            start_time = time.time()
            assert not (distribution is None and CDF is None)
            if n_samples_reference is None:
                self.n_samples_reference = self.n_samples
            else:
                self.n_samples_reference = n_samples_reference
            if n_dimensions_reference is None:
                self.n_dimensions_reference = self.n_dimensions
            else:
                self.n_dimensions_reference = n_dimensions_reference
            if distribution == "uniform":
                uniform_min = reference_distribution_parameters[0]
                uniform_max = reference_distribution_parameters[1]
                self.reference_sample = np.random.uniform(uniform_min, uniform_max, size=(self.n_dimensions_reference, self.n_samples_reference))
            elif distribution == "uniform_manual":
                self.reference_sample = np.zeros((self.n_dimensions_reference, self.n_samples_reference))
                for dimension_ in range(self.n_dimensions_reference):
                    uniform_min_max_bounds = reference_distribution_parameters[dimension_]
                    uniform_min = uniform_min_max_bounds[0]
                    uniform_max = uniform_min_max_bounds[1]
                    self.reference_sample[dimension_, :] = np.random.uniform(uniform_min, uniform_max, size=(1, self.n_samples_reference))
            elif distribution == "blob":
                centers = [reference_distribution_parameters[0]]
                cluster_std = [reference_distribution_parameters[1]]
                self.reference_sample, _ = make_blobs(n_samples=self.n_samples_reference, n_features=self.n_dimensions_reference, centers=centers, cluster_std=cluster_std)
                self.reference_sample = self.reference_sample.T #--> make it column-wise
            elif distribution == "two_blobs":
                centers = reference_distribution_parameters[0]
                cluster_std = reference_distribution_parameters[1]
                self.reference_sample, _ = make_blobs(n_samples=self.n_samples_reference, n_features=self.n_dimensions_reference, centers=centers, cluster_std=cluster_std)
                self.reference_sample = self.reference_sample.T #--> make it column-wise
            elif distribution == "Gaussian":
                mean = reference_distribution_parameters[0]
                cov = reference_distribution_parameters[1]
                self.reference_sample = np.random.multivariate_normal(mean, cov, size=self.n_samples_reference)
                self.reference_sample = self.reference_sample.T #--> make it column-wise
            elif distribution == "s_curve":
                assert self.n_dimensions_reference == 2 or self.n_dimensions_reference == 3
                mean_ = reference_distribution_parameters[0]
                radius_ = reference_distribution_parameters[1]
                X_3d, _ = make_s_curve(n_samples=self.n_samples_reference, random_state=0)
                if self.n_dimensions_reference == 2:
                    self.reference_sample = np.column_stack((X_3d[:, 0], X_3d[:, 2]))
                else:
                    self.reference_sample = X_3d
                self.reference_sample = self.reference_sample.T  # --> make it column-wise
                self.reference_sample *= radius_
                for sample_index in range(self.reference_sample.shape[1]):
                    self.reference_sample[:, sample_index] += mean_
                # self.reference_sample += mean_
            elif distribution == "circle":
                mean_ = reference_distribution_parameters[0]
                radius_ = reference_distribution_parameters[1]
                X_, y_ = make_circles(n_samples=self.n_samples_reference * 2, factor=0.99)  #--> (times 2) because we will exclude the inner circle
                X_ = X_[y_ == 0, :]
                X_ *= radius_
                X_ += mean_
                self.reference_sample = X_.T  # --> make it column-wise
            elif distribution == "swiss_roll":
                mean_ = reference_distribution_parameters[0]
                radius_ = reference_distribution_parameters[1]
                X_3d, _ = make_swiss_roll(n_samples=self.n_samples_reference)
                if self.n_dimensions_reference == 2:
                    self.reference_sample = np.column_stack((X_3d[:, 0], X_3d[:, 2]))
                else:
                    self.reference_sample = X_3d
                self.reference_sample /= 15.0  #--> make scale (radius) almost 1
                self.reference_sample *= radius_
                self.reference_sample += mean_
                self.reference_sample = self.reference_sample.T  # --> make it column-wise
            elif distribution == "thick_circle":
                radius_inner = reference_distribution_parameters[0]
                radius_outer = reference_distribution_parameters[1]
                mean_ = reference_distribution_parameters[2]
                self.reference_sample = np.zeros((self.n_samples_reference, self.n_dimensions_reference))
                for sample_index in range(self.n_samples_reference):
                    while True:
                        sample_dim1 = np.random.uniform(-1 * radius_outer, radius_outer, size=(1, 1))
                        sample_dim2 = np.random.uniform(-1 * radius_outer, radius_outer, size=(1, 1))
                        if (sample_dim1**2 + sample_dim2**2 >= radius_inner) and (sample_dim1**2 + sample_dim2**2 <= radius_outer):
                            self.reference_sample[sample_index, :] = [sample_dim1, sample_dim2]
                            break
                self.reference_sample += mean_
                self.reference_sample = self.reference_sample.T  # --> make it column-wise
            elif distribution == "thick_square":
                radius_inner = reference_distribution_parameters[0]
                radius_outer = reference_distribution_parameters[1]
                mean_ = reference_distribution_parameters[2]
                self.reference_sample = np.zeros((self.n_samples_reference, self.n_dimensions_reference))
                for sample_index in range(self.n_samples_reference):
                    while True:
                        sample_dim1 = np.random.uniform(-1 * radius_outer, radius_outer, size=(1, 1))
                        sample_dim2 = np.random.uniform(-1 * radius_outer, radius_outer, size=(1, 1))
                        if (abs(sample_dim1) >= radius_inner) or (abs(sample_dim2) >= radius_inner):
                            self.reference_sample[sample_index, :] = [sample_dim1, sample_dim2]
                            break
                self.reference_sample += mean_
                self.reference_sample = self.reference_sample.T  # --> make it column-wise
            elif distribution == "triangle":
                mean_ = reference_distribution_parameters[0]
                scale_ = reference_distribution_parameters[1]
                self.reference_sample = np.zeros((self.n_samples_reference, self.n_dimensions_reference))
                for sample_index in range(self.n_samples_reference):
                    while True:
                        sample_dim1 = np.random.uniform(-1, 1, size=(1, 1))
                        sample_dim2 = np.random.uniform(-1, 1, size=(1, 1))
                        if (sample_dim2 <= -3.72*sample_dim1 + 0.86) and (sample_dim2 <= 3.72*sample_dim1 + 0.86):
                            self.reference_sample[sample_index, :] = [sample_dim1, sample_dim2]
                            break
                self.reference_sample *= scale_
                self.reference_sample += mean_
                self.reference_sample = self.reference_sample.T  # --> make it column-wise
            elif distribution == "load_reference_sample":
                self.reference_sample = reference_distribution_parameters
            elif distribution == "CDF":
                self.reference_sample = np.zeros((self.n_dimensions_reference, self.n_samples_reference))
                for sample_index in range(self.n_samples_reference):
                    for dimension_index in range(self.n_dimensions_reference):
                        random_number = random.uniform(0, 1)
                        n_lines = len(reference_distribution_parameters[-1]) #--> the last item in reference_distribution_parameters is the x breaks in inverse CDF
                        for line_index in range(n_lines):
                            if random_number <= reference_distribution_parameters[-1][line_index]:
                                slope_of_line = reference_distribution_parameters[line_index][0]
                                intercept_of_line = reference_distribution_parameters[line_index][1]
                                y_inverse_CDF = (slope_of_line * random_number) + intercept_of_line
                                self.reference_sample[dimension_index, sample_index] = y_inverse_CDF
                                break
            elif distribution == "diamond":
                mean_ = reference_distribution_parameters[0]
                scale_ = reference_distribution_parameters[1]
                self.reference_sample = np.zeros((self.n_samples_reference, self.n_dimensions_reference))
                for sample_index in range(self.n_samples_reference):
                    while True:
                        sample_dim1 = np.random.uniform(-1, 1, size=(1, 1))
                        sample_dim2 = np.random.uniform(-1, 1, size=(1, 1))
                        if (sample_dim2 <= -1*sample_dim1 + 1) and (sample_dim2 <= 1*sample_dim1 + 1) and (sample_dim2 >= 1*sample_dim1 - 1) and (sample_dim2 >= -1*sample_dim1 -1):
                            self.reference_sample[sample_index, :] = [sample_dim1, sample_dim2]
                            break
                self.reference_sample *= scale_
                self.reference_sample += mean_
                self.reference_sample = self.reference_sample.T  # --> make it column-wise
            elif distribution == "CDF_onlyFirstDimension":
                self.reference_sample = np.zeros((self.n_dimensions_reference, self.n_samples_reference))
                for sample_index in range(self.n_samples_reference):
                    for dimension_index in range(self.n_dimensions_reference):
                        if dimension_index == 0:
                            random_number = random.uniform(0, 1)
                            n_lines = len(reference_distribution_parameters[-1]) #--> the last item in reference_distribution_parameters is the x breaks in inverse CDF
                            for line_index in range(n_lines):
                                if random_number <= reference_distribution_parameters[-1][line_index]:
                                    slope_of_line = reference_distribution_parameters[line_index][0]
                                    intercept_of_line = reference_distribution_parameters[line_index][1]
                                    y_inverse_CDF = (slope_of_line * random_number) + intercept_of_line
                                    self.reference_sample[dimension_index, sample_index] = y_inverse_CDF
                                    break
                        else:
                            random_number = random.uniform(0, 1)
                            self.reference_sample[dimension_index, sample_index] = random_number
            end_time = time.time()
            time_make_reference_sample = end_time - start_time
            # self.save_variable(variable=self.reference_sample, name_of_variable="reference_sample", path_to_save=path_save_base)
            # self.save_scatter_of_data(data_=self.reference_sample, data_name="reference_sample", path_save_numpy=path_save_base, path_save_plot=path_save_base, color_map=self.colormap, color_meshgrid=self.reference_sample[0, :], do_plot=self.reference_sample_can_be_plotted)
            self.save_scatter_of_data(data_=self.reference_sample, data_name="reference_sample", path_save_numpy=path_save_base, path_save_plot=path_save_base, color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.reference_sample_can_be_plotted)
            self.save_variable(variable=time_make_reference_sample, name_of_variable="time_make_reference_sample", path_to_save=path_save_base)
            self.save_np_array_to_txt(variable=[time_make_reference_sample], name_of_variable="time_make_reference_sample", path_to_save=path_save_base)
        else:
            self.reference_sample = self.load_variable(name_of_variable="reference_sample", path=path_save_base)
            self.n_samples_reference = self.reference_sample.shape[1]
            self.n_dimensions_reference = self.reference_sample.shape[0]
        if plot_reference_distribution:
            plt.plot(self.reference_sample[0, :], self.reference_sample[1, :], "o")
            plt.show()

    def distribution_transformation_fuzzy_QQplot(self, max_iterations_matching=1000, max_iterations_transformation=1000,
                                                 save_each_how_many_epochs=5, continue_from_previous_run=False, which_iteration_to_load=0, path_save_base="./", match_points_again=True):
        assert self.n_samples_reference == self.n_samples
        if not match_points_again:
            if not os.path.exists(path_save_base+"matched_data/"): #--> if for this class (in supervised case), matching has not been done yet
                match_points_again = True
                continue_from_previous_run = False
        # uncomment this if you want to start again for some classes:
        # if self.which_class_workingOn_now == 0:
        #     continue_from_previous_run = True
        # else:
        #     continue_from_previous_run = False
        if (not continue_from_previous_run) and (match_points_again):
            start_time = time.time()
            if self.reorder_reference_sample_in_matching:
                X_matched, Y_matched = self.match_points_fuzzy_QQplot_reorderReferenceSample(max_iterations=max_iterations_matching, path_to_save=path_save_base+"matched_data/")
            else:
                X_matched, Y_matched = self.match_points_fuzzy_QQplot(max_iterations=max_iterations_matching, path_to_save=path_save_base+"matched_data/")
            # self.save_scatter_of_data(data_=X_matched, data_name="X_matched", path_save_numpy=path_save_base+"matched_data/", path_save_plot=path_save_base+"matched_data/", color_map=self.colormap, color_meshgrid=X_matched[0, :], do_plot=self.dataset_can_be_plotted)
            # self.save_scatter_of_data(data_=Y_matched, data_name="Y_matched", path_save_numpy=path_save_base+"matched_data/", path_save_plot=path_save_base+"matched_data/", color_map=self.colormap, color_meshgrid=X_matched[0, :], do_plot=self.reference_sample_can_be_plotted)
            if not self.supervised_mode:
                self.save_scatter_of_data(data_=X_matched, data_name="X_matched", path_save_numpy=path_save_base+"matched_data/", path_save_plot=path_save_base+"matched_data/", color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.dataset_can_be_plotted)
                self.save_scatter_of_data(data_=Y_matched, data_name="Y_matched", path_save_numpy=path_save_base+"matched_data/", path_save_plot=path_save_base+"matched_data/", color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.reference_sample_can_be_plotted)
            else:
                self.save_scatter_of_data(data_=X_matched, data_name="X_matched", path_save_numpy=path_save_base+"matched_data/", path_save_plot=path_save_base+"matched_data/", color_map=self.colormap, color_meshgrid=X_matched[-1, :], do_plot=self.dataset_can_be_plotted)
                self.save_scatter_of_data(data_=Y_matched, data_name="Y_matched", path_save_numpy=path_save_base+"matched_data/", path_save_plot=path_save_base+"matched_data/", color_map=self.colormap, color_meshgrid=X_matched[-1, :], do_plot=self.reference_sample_can_be_plotted)
            end_time = time.time()
            time_matching = end_time - start_time
            self.save_variable(variable=time_matching, name_of_variable="time_matching", path_to_save=path_save_base+"matched_data/")
            self.save_np_array_to_txt(variable=[time_matching], name_of_variable="time_matching", path_to_save=path_save_base+"matched_data/")
        else:
            # X_matched, Y_matched = None, None
            X_matched = self.load_variable(name_of_variable="X_matched", path=path_save_base+"matched_data/")
            Y_matched = self.load_variable(name_of_variable="Y_matched", path=path_save_base + "matched_data/")
        if self.transform_to_just_shape_of_reference:
            X_matched, Y_matched = self.fit_line_to_QQ_plot(X_matched, Y_matched)
        # X_matched_transformed = self.Quasi_Newton_optimization(X_matched, Y_matched, max_iterations=max_iterations_transformation,
        #                                save_each_how_many_epochs=save_each_how_many_epochs, path_save_base=path_save_base+"fuzzy_QQplot/",
        #                                continue_from_previous_run=continue_from_previous_run, which_iteration_to_load=which_iteration_to_load)
        X_matched_transformed = self.Quasi_Newton_optimization(X_matched, Y_matched, max_iterations=max_iterations_transformation,
                                       save_each_how_many_epochs=5, path_save_base=path_save_base+"fuzzy_QQplot/",
                                       continue_from_previous_run=continue_from_previous_run, which_iteration_to_load=which_iteration_to_load)
        return X_matched_transformed

    def fit_line_to_QQ_plot(self, X_matched, Y_matched):
        n_samples = Y_matched.shape[1]
        n_dimensions = Y_matched.shape[0]
        mu_matrix = np.zeros((n_dimensions, n_samples))
        for dimension_index in range(n_dimensions):
            Y_matched_dimension = np.reshape(Y_matched[dimension_index, :], (-1, 1))
            X_matched_dimension = np.reshape(X_matched[dimension_index, :], (-1, 1))
            Gamma_ = np.column_stack((np.ones((n_samples, 1)), Y_matched_dimension))
            beta_ = np.linalg.inv(Gamma_.T @ Gamma_) @ Gamma_.T @ X_matched_dimension
            mu_ = Gamma_ @ beta_
            mu_matrix[dimension_index, :] = np.reshape(mu_, (1, -1))
        Y_matched = mu_matrix
        return X_matched, Y_matched

    def Quasi_Newton_optimization(self, X_matched, Y_matched, max_iterations=100, save_each_how_many_epochs=5, plot_the_last_solution=False, path_save_base="./algorithm_files/", continue_from_previous_run=False, which_iteration_to_load=0):
        if not continue_from_previous_run:
            X_matched_initial = X_matched.copy()
            # self.save_scatter_of_data(data_=X_matched_initial, data_name="X_matched_initial", path_save_numpy=path_save_base, path_save_plot=path_save_base, color_map=self.colormap, color_meshgrid=X_matched_initial[0, :], do_plot=self.dataset_can_be_plotted)
            # self.save_scatter_of_data(data_=Y_matched, data_name="Y_matched", path_save_numpy=path_save_base, path_save_plot=path_save_base, color_map=self.colormap, color_meshgrid=X_matched_initial[0, :], do_plot=self.reference_sample_can_be_plotted)
            # self.save_scatter_of_data(data_=X_matched_initial, data_name="X_matched_initial", path_save_numpy=path_save_base, path_save_plot=path_save_base, color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.dataset_can_be_plotted)
            # self.save_scatter_of_data(data_=Y_matched, data_name="Y_matched", path_save_numpy=path_save_base, path_save_plot=path_save_base, color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.reference_sample_can_be_plotted)
            self.save_scatter_of_data_2(data_=X_matched_initial, data_name="X_matched_initial", path_save_numpy=path_save_base, path_save_plot=path_save_base, path_save_plot_images=path_save_base+"/initial_images/", color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.dataset_can_be_plotted, plot_embedding_of_images=self.plot_embedding_of_images)
            self.save_scatter_of_data_2(data_=Y_matched, data_name="Y_matched", path_save_numpy=path_save_base, path_save_plot=path_save_base, path_save_plot_images=None, color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.reference_sample_can_be_plotted, plot_embedding_of_images=False)
            iteration_start = 0
            objective_function_toSave, objective_function_QQ_part_toSave, objective_function_distance_part_toSave = [], [], []
            time_quasi_newton = []
        else:
            X_matched_initial = self.load_variable(name_of_variable="X_matched_initial", path=path_save_base)
            Y_matched = self.load_variable(name_of_variable="Y_matched", path=path_save_base)
            X_matched = self.load_variable(name_of_variable="X_matched_iteration_"+str(which_iteration_to_load), path=path_save_base+"iterations_numpy/")
            iteration_start = which_iteration_to_load
            objective_function_toSave = self.load_variable(name_of_variable="objective_function", path=path_save_base)
            objective_function_QQ_part_toSave = self.load_variable(name_of_variable="objective_function_QQ_part_toSave", path=path_save_base)
            objective_function_distance_part_toSave = self.load_variable(name_of_variable="objective_function_distance_part_toSave", path=path_save_base)
            time_quasi_newton = self.load_variable(name_of_variable="time_quasi_newton", path=path_save_base)
            objective_function_toSave = list(objective_function_toSave)
            objective_function_QQ_part_toSave = list(objective_function_QQ_part_toSave)
            objective_function_distance_part_toSave = list(objective_function_distance_part_toSave)
            time_quasi_newton = list(time_quasi_newton)
            objective_function_toSave = objective_function_toSave[:iteration_start]
            objective_function_QQ_part_toSave = objective_function_QQ_part_toSave[:iteration_start]
            objective_function_distance_part_toSave = objective_function_distance_part_toSave[:iteration_start]
            time_quasi_newton = time_quasi_newton[:iteration_start]
        if self.save_umap_of_iterations:
            self.save_umap_of_data(data_=X_matched_initial, data_name="X_matched_initial", path_save_plots=path_save_base+"iterations_umap/")
        KNN_distance_matrix_initial, neighbors_indices = self.find_KNN_distance_matrix(X=X_matched_initial, n_neighbors=self.n_neighbors)
        normalization_factor = sum(sum(KNN_distance_matrix_initial))
        start_time = time.time()
        for iteration_index in range(iteration_start, max_iterations):
            print("Performing quasi Newton, iteration " + str(iteration_index))
            All_NN_distance_matrix, _ = self.find_KNN_distance_matrix(X=X_matched, n_neighbors=self.n_samples-1)
            for sample_index in range(self.n_samples):
                for dimension_index in range(self.n_dimensions_reference):
                    # --- calculate gradient and second derivative of gradient (Hessian):
                    gradient_term1 = X_matched[dimension_index, sample_index] - Y_matched[dimension_index, sample_index]
                    gradient_term2 = 0.0
                    Hessian_term2 = 0.0
                    for neighbor_index in range(self.n_neighbors):
                        neighbor_index_in_dataset = neighbors_indices[sample_index, neighbor_index]
                        d = All_NN_distance_matrix[sample_index, neighbor_index_in_dataset]
                        d_initial = KNN_distance_matrix_initial[sample_index, neighbor_index_in_dataset]
                        gradient_term2 += ((d - d_initial) / (d * d_initial)) * (X_matched[dimension_index, sample_index] - X_matched[dimension_index, neighbor_index_in_dataset])
                        Hessian_term2 += ((d - d_initial) / (d * d_initial)) - ((X_matched[dimension_index, sample_index] - X_matched[dimension_index, neighbor_index_in_dataset])**2 / d**3)
                    gradient_term2 *= (1 / normalization_factor)
                    Hessian_term2 *= (1 / normalization_factor)
                    gradient_ = (self.regularization_parameter * gradient_term1) + gradient_term2
                    Hessian_ = (1 * self.regularization_parameter) + Hessian_term2
                    # --- update solution:
                    X_matched[dimension_index, sample_index] = X_matched[dimension_index, sample_index] - (self.learning_rate_fuzzy_QQplot * abs(1 / Hessian_) * gradient_)
            # calculate the objective function:
            objective_function_QQ_part = 0.0
            objective_function_distance_part = 0.0
            for sample_index in range(self.n_samples):
                objective_function_QQ_part += np.linalg.norm(X_matched[:, sample_index] - Y_matched[:, sample_index])**2
                temp_ = 0.0
                for neighbor_index in range(self.n_neighbors):
                    neighbor_index_in_dataset = neighbors_indices[sample_index, neighbor_index]
                    d = All_NN_distance_matrix[sample_index, neighbor_index_in_dataset]
                    d_initial = KNN_distance_matrix_initial[sample_index, neighbor_index_in_dataset]
                    temp_ += (d - d_initial)**2 / d_initial
                objective_function_distance_part += (1 / normalization_factor) * temp_
            objective_function = 0.5 * ((self.regularization_parameter * objective_function_QQ_part) + objective_function_distance_part)
            objective_function_toSave.append(objective_function)
            objective_function_QQ_part_toSave.append(objective_function_QQ_part)
            objective_function_distance_part_toSave.append(objective_function_distance_part)
            print("iteration " + str(iteration_index) + ": total cost = " + str(objective_function) +
                  ", QQ cost = " + str(objective_function_QQ_part) + ", distance cost = " + str(objective_function_distance_part))
            end_time = time.time()
            time_quasi_newton.append(end_time - start_time)
            if (iteration_index % save_each_how_many_epochs) == 0:
                # self.save_scatter_of_data(data_=X_matched, data_name="X_matched_iteration_"+str(iteration_index), path_save_numpy=path_save_base+"iterations_numpy/", path_save_plot=path_save_base+"iterations_plot/", color_map=self.colormap, color_meshgrid=X_matched_initial[0, :], do_plot=self.dataset_can_be_plotted)
                # X_matched_outliersRemoved, color_meshgrid_outliersRemoved = self.remove_outliers(data_=X_matched, color_meshgrid=X_matched_initial[0, :])
                # self.save_scatter_of_data(data_=X_matched, data_name="X_matched_iteration_"+str(iteration_index), path_save_numpy=path_save_base+"iterations_numpy/", path_save_plot=path_save_base+"iterations_plot/", color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.dataset_can_be_plotted)
                if (abs(iteration_index - max_iterations) <= save_each_how_many_epochs) and self.plot_embedding_of_images:
                    plot_embedding_of_images = True
                else:
                    plot_embedding_of_images = False
                self.save_scatter_of_data_2(data_=X_matched, data_name="X_matched_iteration_"+str(iteration_index), path_save_numpy=path_save_base+"iterations_numpy/", path_save_plot=path_save_base+"iterations_plot/", path_save_plot_images=path_save_base+"iterations_plot_2/", color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.dataset_can_be_plotted, plot_embedding_of_images=plot_embedding_of_images)
                # if not self.supervised_mode:
                #     X_matched_outliersRemoved, color_meshgrid_outliersRemoved = self.remove_outliers(data_=X_matched, color_meshgrid=self.color_meshgrid)
                #     self.save_scatter_of_data(data_=X_matched_outliersRemoved, data_name="X_matched_iteration_"+str(iteration_index), path_save_numpy=path_save_base+"iterations_numpy_noOutliers/", path_save_plot=path_save_base+"iterations_plot_noOutliers/", color_map=self.colormap, color_meshgrid=color_meshgrid_outliersRemoved, do_plot=self.dataset_can_be_plotted)
                # else:
                #     self.save_scatter_of_data_3_removingOutliersForSupervised(data_=X_matched, data_name="X_matched_iteration_"+str(iteration_index), path_save_numpy=path_save_base+"iterations_numpy_noOutliers/", path_save_plot=path_save_base+"iterations_plot_noOutliers/", color_map=self.colormap, color_meshgrid=self.color_meshgrid, do_plot=self.dataset_can_be_plotted)
                if self.save_images_of_iterations:
                    self.save_image_of_data(data_=X_matched, data_name="X_matched_iteration_"+str(iteration_index), path_save_images=path_save_base+"iterations_images/")
                if self.save_umap_of_iterations:
                    self.save_umap_of_data(data_=X_matched, data_name="X_matched_iteration_"+str(iteration_index), path_save_plots=path_save_base+"iterations_umap/")
                self.save_variable(variable=np.asarray(objective_function_toSave), name_of_variable="objective_function", path_to_save=path_save_base)
                self.save_np_array_to_txt(variable=np.column_stack((np.array([i for i in range(iteration_index+1)]).T, np.asarray(objective_function_toSave).T)), name_of_variable="objective_function", path_to_save=path_save_base)
                self.save_variable(variable=np.asarray(objective_function_QQ_part_toSave), name_of_variable="objective_function_QQ_part_toSave", path_to_save=path_save_base)
                self.save_np_array_to_txt(variable=np.column_stack((np.array([i for i in range(iteration_index+1)]).T, np.asarray(objective_function_QQ_part_toSave).T)), name_of_variable="objective_function_QQ_part_toSave", path_to_save=path_save_base)
                self.save_variable(variable=np.asarray(objective_function_distance_part_toSave), name_of_variable="objective_function_distance_part_toSave", path_to_save=path_save_base)
                self.save_np_array_to_txt(variable=np.column_stack((np.array([i for i in range(iteration_index+1)]).T, np.asarray(objective_function_distance_part_toSave).T)), name_of_variable="objective_function_distance_part_toSave", path_to_save=path_save_base)
                self.save_variable(variable=time_quasi_newton, name_of_variable="time_quasi_newton", path_to_save=path_save_base)
                self.save_np_array_to_txt(variable=np.column_stack((np.array([i for i in range(iteration_index+1)]).T, np.asarray(time_quasi_newton).T)), name_of_variable="time_quasi_newton", path_to_save=path_save_base)
        # plot the points:
        if plot_the_last_solution:
            cm_bright = self.colormap
            # color_meshgrid = X_matched_initial[0, :]
            color_meshgrid = self.color_meshgrid
            plt.subplot(1, 3, 1)
            plt.scatter(Y_matched[0, :], Y_matched[1, :], c=color_meshgrid, cmap=cm_bright, edgecolors='k')
            plt.subplot(1, 3, 2)
            plt.scatter(X_matched_initial[0, :], X_matched_initial[1, :], c=color_meshgrid, cmap=cm_bright, edgecolors='k')
            plt.subplot(1, 3, 3)
            plt.scatter(X_matched[0, :], X_matched[1, :], c=color_meshgrid, cmap=cm_bright, edgecolors='k')
            plt.show()
        return X_matched

    def remove_outliers(self, data_, color_meshgrid):
        # data_: column-wise samples
        data_outliers_removed = data_.copy()
        color_meshgrid_outliers_removed = color_meshgrid.copy()
        for dimension_index in range(data_.shape[0]):
            data_dimension = data_[dimension_index, :].ravel()
            # Set upper and lower limit to 3 standard deviation
            data_dimension_std = np.std(data_dimension)
            data_dimension_mean = np.mean(data_dimension)
            anomaly_cut_off = data_dimension_std * 3
            lower_limit = data_dimension_mean - anomaly_cut_off
            upper_limit = data_dimension_mean + anomaly_cut_off
            samples_to_keep = []
            for sample_index in range(data_outliers_removed.shape[1]):
                sample_ = data_outliers_removed[:, sample_index]
                if sample_[dimension_index] > upper_limit or sample_[dimension_index] < lower_limit:
                    samples_to_keep.append(False)
                else:
                    samples_to_keep.append(True)
            data_outliers_removed = data_outliers_removed.compress(samples_to_keep, axis=1)
            color_meshgrid_outliers_removed = color_meshgrid_outliers_removed.compress(samples_to_keep)
        return data_outliers_removed, color_meshgrid_outliers_removed

    def remove_outliers_2(self, data_, color_meshgrid):
        # data_: column-wise samples
        indices_of_points_in_theClassWorkingOn = np.asarray(self.indices_of_points_in_classes[self.which_class_workingOn_now]).copy()
        data_outliers_removed = data_.copy()
        color_meshgrid_outliers_removed = color_meshgrid.copy()
        indices_of_points_in_theClassWorkingOn_outliers = []
        samples_to_remove = []
        for dimension_index in range(data_.shape[0]):
            data_dimension = data_[dimension_index, :].ravel()
            # Set upper and lower limit to 3 standard deviation
            data_dimension_std = np.std(data_dimension)
            data_dimension_mean = np.mean(data_dimension)
            anomaly_cut_off = data_dimension_std * 3
            lower_limit = data_dimension_mean - anomaly_cut_off
            upper_limit = data_dimension_mean + anomaly_cut_off
            # samples_to_keep = []
            for sample_index in range(data_.shape[1]):
                sample_ = data_outliers_removed[:, sample_index]
                if sample_[dimension_index] > upper_limit or sample_[dimension_index] < lower_limit:
                    # samples_to_keep.append(False)
                    samples_to_remove.append(sample_index)
                    indices_of_points_in_theClassWorkingOn_outliers.append(indices_of_points_in_theClassWorkingOn[sample_index])
        indices_of_points_in_theClassWorkingOn_outliers_removed = [indices_of_points_in_theClassWorkingOn[i] for i in range(data_.shape[1]) if i not in samples_to_remove]
        samples_to_keep = [i for i in range(data_.shape[1]) if i not in samples_to_remove]
        data_outliers_removed = np.zeros((data_.shape[0], data_.shape[1]-len(samples_to_remove)))
        counter_ = -1
        for sample_index in range(data_.shape[1]):
            if sample_index not in samples_to_remove:
                counter_ += 1
                if counter_ < data_outliers_removed.shape[1]:
                    data_outliers_removed[:, counter_] = data_[:, sample_index]
        color_meshgrid_outliers_removed = color_meshgrid_outliers_removed.compress(samples_to_keep)
        return data_outliers_removed, color_meshgrid_outliers_removed, \
               indices_of_points_in_theClassWorkingOn_outliers_removed, indices_of_points_in_theClassWorkingOn_outliers

    def save_scatter_of_data(self, data_, data_name, path_save_numpy, path_save_plot, color_map, color_meshgrid, do_plot=True):
        self.save_variable(variable=data_, name_of_variable=data_name, path_to_save=path_save_numpy)
        if do_plot:
            fig, ax = plt.subplots()
            if not os.path.exists(path_save_plot):
                os.makedirs(path_save_plot)
            plt.scatter(data_[0, :], data_[1, :], c=color_meshgrid, cmap=color_map, edgecolors='k')
            # plt.show()
            plt.savefig(path_save_plot + data_name + ".png")
            plt.clf()
            # plt.close()
            self.save_variable(variable=data_, name_of_variable=data_name+"_plotData", path_to_save=path_save_plot + "figs/")
            self.save_variable(variable=color_meshgrid, name_of_variable=data_name + "_plotColors", path_to_save=path_save_plot + "figs/")

    def save_image_of_data(self, data_, data_name, path_save_images, scale=2):
        if not os.path.exists(path_save_images+data_name+"/"):
            os.makedirs(path_save_images+data_name+"/")
        for sample_index in range(data_.shape[1]):
            sample = data_[:, sample_index].reshape((self.image_height, self.image_width))
            sample = resize(sample, (int(sample.shape[0]*scale), int(sample.shape[1]*scale)), order=5, preserve_range=True, mode="constant")
            im = Image.fromarray(sample)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(path_save_images+data_name+"/"+str(sample_index)+".png")

    def save_umap_of_data(self, data_, data_name, path_save_plots):
        if not os.path.exists(path_save_plots):
            os.makedirs(path_save_plots)
        if self.supervised_mode:
            X_all_classes = self.X_multiclass_to_plot.copy()
            X_all_classes[:, self.indices_of_points_in_classes[self.which_class_workingOn_now]] = data_
            data_all = X_all_classes
            # labels_all = self.y
            labels_all = self.labels_of_all_dataset
        else:
            labels_all = np.zeros((data_.shape[1],))
            data_all = data_
        if data_all.shape[0] == 2:
            data_dimension_reduced = data_all
        else:
            data_dimension_reduced = (umap.UMAP(n_neighbors=500).fit_transform(data_all.T)).T
        # _, ax = plt.subplots(1, figsize=(14, 10))
        _, ax = plt.subplots(1, figsize=(5, 5))
        n_classes = len(np.unique(labels_all))
        classes = [str(i) for i in range(n_classes)]
        # plt.scatter(data_sampled[0, :], data_sampled[1, :], s=10, c=labels_sampled, cmap=self.colormap, alpha=1.0)
        markers = ["v", "o"]
        colors = ["r", "b"]
        for class_index in range(n_classes):
            sample_of_this_class = data_dimension_reduced[:, labels_all == class_index]
            # c = class_index * np.ones((sample_of_this_class.shape[1],))
            plt.scatter(sample_of_this_class[0, :], sample_of_this_class[1, :], s=30, color=colors[class_index], alpha=1.0, marker=markers[class_index])
        # cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        # cbar.set_ticks(np.arange(n_classes))
        # cbar.set_ticklabels(classes)
        # plt.show()
        plt.savefig(path_save_plots + data_name + ".png")
        plt.clf()
        # plt.close()
        self.save_variable(variable=data_dimension_reduced, name_of_variable=data_name+"_plotData", path_to_save=path_save_plots + "figs/")
        self.save_variable(variable=labels_all, name_of_variable=data_name + "_plotColors", path_to_save=path_save_plots + "figs/")
        plot_embedding_of_images = True
        if plot_embedding_of_images:
            if not os.path.exists(path_save_plots + "images/"):
                os.makedirs(path_save_plots + "images/")
            # self.plot_embedding(X=data_, image_scale=0.2,  title=None)
            n_samples = self.X_images_to_plot.shape[1]
            scale = 1
            dataset_notReshaped = np.zeros((n_samples, self.image_height*scale, self.image_width*scale))
            for image_index in range(n_samples):
                image = self.X_images_to_plot[:, image_index]
                image_not_reshaped = image.reshape((self.image_height, self.image_width))
                image_not_reshaped_scaled = image_not_reshaped
                dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
            images = dataset_notReshaped
            # fig, ax = plt.subplots(figsize=(10, 10))
            # self.plot_components(X_projected=data_dimension_reduced, which_dimensions_to_plot=[0,1], images=images, ax=None, image_scale=0.4, markersize=10, thumb_frac=0.08, cmap="gray")
            self.plot_components(X_projected=data_dimension_reduced, which_dimensions_to_plot=[0,1], images=images, ax=None, image_scale=0.8, markersize=10, thumb_frac=0.03, cmap="gray")
            plt.savefig(path_save_plots + "images/" + data_name + ".png")
            plt.clf()
            plt.close()

    def save_scatter_of_data_2(self, data_, data_name, path_save_numpy, path_save_plot, path_save_plot_images, color_map, color_meshgrid, do_plot=True, plot_embedding_of_images=False):
        self.save_variable(variable=data_, name_of_variable=data_name, path_to_save=path_save_numpy)
        if do_plot:
            fig, ax = plt.subplots()
            if not os.path.exists(path_save_plot):
                os.makedirs(path_save_plot)
            if not self.supervised_mode:
                plt.scatter(data_[0, :], data_[1, :], c=color_meshgrid, cmap=color_map, edgecolors='k')
            else:
                self.X_multiclass_to_plot[:, self.indices_of_points_in_classes[self.which_class_workingOn_now]] = data_
                plt.scatter(self.X_multiclass_to_plot[0, :], self.X_multiclass_to_plot[1, :], c=self.labels_of_all_dataset, cmap=color_map, edgecolors='k')
                data_ = self.X_multiclass_to_plot
            # plt.show()
            plt.savefig(path_save_plot + data_name + ".png")
            plt.clf()
            # plt.close()
            self.save_variable(variable=data_, name_of_variable=data_name+"_plotData", path_to_save=path_save_plot + "figs/")
            self.save_variable(variable=color_meshgrid, name_of_variable=data_name + "_plotColors", path_to_save=path_save_plot + "figs/")
            if plot_embedding_of_images:
                if not os.path.exists(path_save_plot_images):
                    os.makedirs(path_save_plot_images)
                # self.plot_embedding(X=data_, image_scale=0.2,  title=None)
                n_samples = self.X_images_to_plot.shape[1]
                scale = 1
                dataset_notReshaped = np.zeros((n_samples, self.image_height*scale, self.image_width*scale))
                for image_index in range(n_samples):
                    image = self.X_images_to_plot[:, image_index]
                    image_not_reshaped = image.reshape((self.image_height, self.image_width))
                    image_not_reshaped_scaled = image_not_reshaped
                    dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
                images = dataset_notReshaped
                # fig, ax = plt.subplots(figsize=(10, 10))
                # self.plot_components(X_projected=data_, which_dimensions_to_plot=[0,1], images=images, ax=None, image_scale=0.4, markersize=10, thumb_frac=0.08, cmap="gray")
                self.plot_components(X_projected=data_, which_dimensions_to_plot=[0,1], images=images, ax=None, image_scale=0.8, markersize=10, thumb_frac=0.03, cmap="gray")
                plt.savefig(path_save_plot_images + data_name + ".png")
                plt.clf()
                plt.close()

    def save_scatter_of_data_3_removingOutliersForSupervised(self, data_, data_name, path_save_numpy, path_save_plot, color_map, color_meshgrid, do_plot=True):
        data_outliers_removed, color_meshgrid_outliers_removed, \
        indices_of_points_in_theClassWorkingOn_outliers_removed, indices_of_points_in_theClassWorkingOn_outliers = self.remove_outliers_2(data_=data_, color_meshgrid=color_meshgrid)
        self.save_variable(variable=data_outliers_removed, name_of_variable=data_name, path_to_save=path_save_numpy)
        if do_plot:
            if not os.path.exists(path_save_plot):
                os.makedirs(path_save_plot)
            X_multiclass_to_plot = self.X_multiclass_to_plot.copy()
            if len(indices_of_points_in_theClassWorkingOn_outliers_removed) != data_outliers_removed.shape[1]:
                return
            X_multiclass_to_plot[:, indices_of_points_in_theClassWorkingOn_outliers_removed] = data_outliers_removed
            X_multiclass_to_plot_outliersRemoved = np.zeros((X_multiclass_to_plot.shape[0], X_multiclass_to_plot.shape[1]-len(indices_of_points_in_theClassWorkingOn_outliers)))
            labels_of_kept_points = np.zeros((X_multiclass_to_plot.shape[1]-len(indices_of_points_in_theClassWorkingOn_outliers),))
            counter_ = -1
            for sample_index in range(X_multiclass_to_plot.shape[1]):
                if sample_index not in indices_of_points_in_theClassWorkingOn_outliers:
                    counter_ += 1
                    X_multiclass_to_plot_outliersRemoved[:, counter_] = X_multiclass_to_plot[:, sample_index]
                    labels_of_kept_points[counter_] = self.labels_of_all_dataset[sample_index]
            plt.scatter(X_multiclass_to_plot_outliersRemoved[0, :], X_multiclass_to_plot_outliersRemoved[1, :], c=labels_of_kept_points, cmap=color_map, edgecolors='k')
            # plt.show()
            plt.savefig(path_save_plot + data_name + ".png")
            plt.clf()
            # plt.close()
            self.save_variable(variable=X_multiclass_to_plot_outliersRemoved, name_of_variable=data_name+"_plotData", path_to_save=path_save_plot + "figs/")
            self.save_variable(variable=X_multiclass_to_plot_outliersRemoved, name_of_variable=data_name + "_plotColors", path_to_save=path_save_plot + "figs/")

    def find_KNN_distance_matrix(self, X, n_neighbors):
        # X: column-wise samples
        # returns KNN_distance_matrix: row-wise --> shape: (n_samples, n_samples) where zero for not neighbors
        # returns neighbors_indices: row-wise --> shape: (n_samples, n_neighbors)
        knn = KNN(n_neighbors=n_neighbors+1, algorithm='kd_tree', n_jobs=-1)  #+1 because the point itself is also counted
        knn.fit(X=X.T)
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph
        # the following function gives n_samples*n_samples matrix, and puts 0 for diagonal and also where points are not connected directly in KNN graph
        # if K=n_samples, only diagonal is zero.
        Euclidean_distance_matrix = knn.kneighbors_graph(X=X.T, n_neighbors=n_neighbors+1, mode='distance') #--> gives Euclidean distances
        KNN_distance_matrix = Euclidean_distance_matrix.toarray()
        neighbors_indices = np.zeros((KNN_distance_matrix.shape[0], n_neighbors))
        for sample_index in range(KNN_distance_matrix.shape[0]):
            neighbors_indices[sample_index, :] = np.ravel(np.asarray(np.where(KNN_distance_matrix[sample_index, :] != 0)))
        neighbors_indices = neighbors_indices.astype(int)
        return KNN_distance_matrix, neighbors_indices

    def match_points_fuzzy_QQplot(self, max_iterations=100, path_to_save="./"):
        # A = np.random.rand(self.n_dimensions_reference, self.n_dimensions)  # --> rand in [0,1)
        A = np.eye(self.n_dimensions_reference)
        b = np.random.rand(self.n_dimensions_reference, 1)  # --> rand in [0,1)
        start_time = time.time()
        time_mathing_iterations = []
        for iteration_index in range(max_iterations):
            print("matching points in fuzzy QQ-plot: iteration " + str(iteration_index) + "...")
            X_resorted = self.assignment_problem(A, b)
            X_hat = np.column_stack(( X_resorted.T, np.ones((self.n_samples, 1)) ))
            Y_hat = self.reference_sample.T
            beta_ = np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ Y_hat
            beta_transpose = beta_.T
            A = beta_transpose[:, :-1]
            b = beta_transpose[:, -1]
            end_time = time.time()
            time_mathing_iterations.append(end_time - start_time)
        self.save_variable(variable=time_mathing_iterations, name_of_variable="time_mathing_iterations", path_to_save=path_to_save)
        self.save_np_array_to_txt(variable=np.column_stack((np.array([i for i in range(max_iterations)]).T, np.asarray(time_mathing_iterations).T)), name_of_variable="time_mathing_iterations", path_to_save=path_to_save)
        # cm_bright = self.colormap
        # color_meshgrid = self.reference_sample[0, :]
        # plt.subplot(1, 2, 1)
        # plt.scatter(self.reference_sample[0, :], self.reference_sample[1, :], c=color_meshgrid, cmap=cm_bright, edgecolors='k')
        # plt.subplot(1, 2, 2)
        # plt.scatter(X_resorted[0, :], X_resorted[1, :], c=color_meshgrid, cmap=cm_bright, edgecolors='k')
        # plt.show()
        X_matched = X_resorted
        Y_matched = self.reference_sample
        return X_matched, Y_matched

    def assignment_problem(self, A, b):
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        cost_matrix = np.zeros((self.n_samples_reference, self.n_samples))
        for row_index in range(self.n_samples_reference):
            for col_index in range(self.n_samples):
                cost_matrix[row_index, col_index] = (np.linalg.norm(self.reference_sample[:, row_index] - (A @ self.X[:, col_index]) - b)) ** 2
        row_ind_assignment, col_ind_assignment = linear_sum_assignment(cost_matrix)
        # resort the sample X according to assignment:
        X_resorted = np.zeros((self.n_dimensions, self.n_samples))
        for sorted_sample_index in range(self.n_samples):
            X_resorted[:, sorted_sample_index] = self.X[:, col_ind_assignment[sorted_sample_index]]
        return X_resorted

    def match_points_fuzzy_QQplot_reorderReferenceSample(self, max_iterations=100, path_to_save="./"):
        # A = np.random.rand(self.n_dimensions_reference, self.n_dimensions)  # --> rand in [0,1)
        A = np.eye(self.n_dimensions)
        b = np.random.rand(self.n_dimensions, 1)  # --> rand in [0,1)
        start_time = time.time()
        time_mathing_iterations = []
        for iteration_index in range(max_iterations):
            print("matching points in fuzzy QQ-plot: iteration " + str(iteration_index) + "...")
            Y_resorted = self.assignment_problem_reorderReferenceSample(A, b)
            Y_hat = np.column_stack(( Y_resorted.T, np.ones((self.n_samples_reference, 1)) ))
            X_hat = self.X.T
            beta_ = np.linalg.inv(Y_hat.T @ Y_hat) @ Y_hat.T @ X_hat
            beta_transpose = beta_.T
            A = beta_transpose[:, :-1]
            b = beta_transpose[:, -1]
            end_time = time.time()
            time_mathing_iterations.append(end_time - start_time)
        self.save_variable(variable=time_mathing_iterations, name_of_variable="time_mathing_iterations", path_to_save=path_to_save)
        self.save_np_array_to_txt(variable=np.column_stack((np.array([i for i in range(max_iterations)]).T, np.asarray(time_mathing_iterations).T)), name_of_variable="time_mathing_iterations", path_to_save=path_to_save)
        Y_matched = Y_resorted
        X_matched = self.X
        return X_matched, Y_matched

    def assignment_problem_reorderReferenceSample(self, A, b):
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        cost_matrix = np.zeros((self.n_samples, self.n_samples_reference))
        for row_index in range(self.n_samples):
            if row_index % 10 == 0:
                print("row index="+str(row_index)+"/"+str(self.n_samples))
            for col_index in range(self.n_samples_reference):
                # print("row index="+str(row_index)+"/"+str(self.n_samples)+", col index="+str(col_index)+"/"+str(self.n_samples_reference))
                cost_matrix[row_index, col_index] = (np.linalg.norm(self.X[:, row_index] - (A @ self.reference_sample[:, col_index]) - b)) ** 2
        row_ind_assignment, col_ind_assignment = linear_sum_assignment(cost_matrix)
        # resort the reference sample according to assignment:
        Y_resorted = np.zeros((self.n_dimensions_reference, self.n_samples_reference))
        for sorted_sample_index in range(self.n_samples):
            Y_resorted[:, sorted_sample_index] = self.reference_sample[:, col_ind_assignment[sorted_sample_index]]
        return Y_resorted

    def separate_samples_of_classes_2(self, X, y):  # it does not change the order of the samples within every class
        # X --> rows: features, columns: samples
        # return X_separated_classes --> each element of list --> rows: features, columns: samples
        y = np.asarray(y)
        y = y.reshape((-1, 1)).ravel()
        labels_of_classes = sorted(set(y.ravel().tolist()))
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        n_classes = len(labels_of_classes)
        X_separated_classes = [np.empty((n_dimensions, 0))] * n_classes
        original_index_in_whole_dataset = [None] * n_classes
        for class_index in range(n_classes):
            original_index_in_whole_dataset[class_index] = []
            for sample_index in range(n_samples):
                if y[sample_index] == labels_of_classes[class_index]:
                    X_separated_classes[class_index] = np.column_stack((X_separated_classes[class_index], X[:, sample_index].reshape((-1,1))))
                    original_index_in_whole_dataset[class_index].append(sample_index)
        return X_separated_classes, original_index_in_whole_dataset

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))

    ###### this function is not used in the code:
    def plot_embedding(self, X, image_scale=None, title=None):
        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
        # X: rows are features, columns are samples
        # y: a row vector (vector of labels)
        n_samples = X.shape[1]
        scale = 1
        image_height = 112
        image_width = 92
        dataset_notReshaped = np.zeros((n_samples, image_height*scale, image_width*scale))
        for image_index in range(n_samples):
            image = self.X_before_dimension_reduction[:, image_index]
            image_not_reshaped = image.reshape((image_height, image_width))
            image_not_reshaped_scaled = image_not_reshaped
            dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
        # fig, ax = plt.subplots(figsize=(10, 10))
        images = dataset_notReshaped
        X = X.T
        # y = y.ravel()
        # y = y.astype(int)
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        # plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(self.y[i]),
                     color=plt.cm.Set1(self.y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
            # plt.text(X[i, 0], X[i, 1], " ",
            #          color=self.color_meshgrid,
            #          fontdict={'weight': 'bold', 'size': 9})
        if image_scale != None:
            images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_components(self, X_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
        # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
        # X_projected: rows are features and columns are samples
        # which_dimensions_to_plot: a list of two integers, index starting from 0
        X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
        X_projected = X_projected.T
        # ax = ax or plt.gca()
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
        ax.scatter(X_projected[:, 0], X_projected[:, 1], c=self.labels_of_all_dataset, cmap=self.colormap, edgecolors='k')
        images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
        if images is not None:
            min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
            shown_images = np.array([2 * X_projected.max(0)])
            for i in range(X_projected.shape[0]):
                dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # don't show points that are too close
                    continue
                shown_images = np.vstack([shown_images, X_projected[i]])
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
                ax.add_artist(imagebox)
            # plot the first (original) image once more to be on top of other images:
            # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
            # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
            # ax.add_artist(imagebox)
        plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
        plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
        plt.xticks([])
        plt.yticks([])
        # plt.show()
