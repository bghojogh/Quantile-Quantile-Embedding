
import tensorflow as tf
import os
import glob
import numpy as np
# from PIL import Image
import umap
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


class Target_domain():

    def __init__(self, checkpoint_dir, model_dir_, model_name, batch_size, feature_space_dimension):
        self.batch_size = batch_size
        self.n_samples = None
        self.n_batches = None


        # X, embedding: rows are samples, columns are features
        # self.saver = tf.train.Saver()
        self.checkpoint_dir = checkpoint_dir
        self.model_dir_ = model_dir_
        
        # self.model_name = model_name
        # self.batch_size = batch_size
        # self.X = X
        # self.n_samples = self.X.shape[0]
        self.feature_space_dimension = feature_space_dimension
        pass

    def embed_data_in_the_source_domain(self, batches, batches_subtypes, siamese, path_save_embeddings_of_test_data):
        print("Embedding the test set into the source domain....")
        n_batches = int(np.ceil(self.n_samples / self.batch_size))
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            embedding = np.zeros((self.n_samples, self.feature_space_dimension))
            subtypes = [None] * self.n_samples
            for batch_index in range(n_batches):
                print("processing batch " + str(batch_index) + "/" + str(n_batches-1))
                X_batch = batches[batch_index]
                could_load, checkpoint_counter = self.load_network_model(sess)
                if could_load:
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")
                X_batch = self.normalize_images(X_batch)
                test_feed_dict = {
                    siamese.x1: X_batch
                }
                embedding_batch = sess.run(siamese.o1, feed_dict=test_feed_dict)
                pass
                if batch_index != (n_batches-1):
                    embedding[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size), :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size)] = batches_subtypes[batch_index]
                else:
                    embedding[(batch_index * self.batch_size) : , :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ] = batches_subtypes[batch_index]
            if not os.path.exists(path_save_embeddings_of_test_data+"numpy\\"):
                os.makedirs(path_save_embeddings_of_test_data+"numpy\\")
            np.save(path_save_embeddings_of_test_data+"numpy\\embedding.npy", embedding)
            np.save(path_save_embeddings_of_test_data+"numpy\\subtypes.npy", subtypes)
            if not os.path.exists(path_save_embeddings_of_test_data+"plots\\"):
                os.makedirs(path_save_embeddings_of_test_data+"plots\\")
            # plt.figure(200)
            plt = self.Kather_get_color_and_shape_of_points(embedding=embedding, subtype_=subtypes)
            plt.savefig(path_save_embeddings_of_test_data+"plots\\" + 'embedding.png')
            plt.clf()
            plt.close()
        return embedding, subtypes

    def normalize_images(self, X_batch):
        # also see normalize_images() method in Utils.py
        X_batch = X_batch * (1. / 255) - 0.5 
        return X_batch

    def Kather_get_color_and_shape_of_points(self, embedding, subtype_):
        if embedding.shape[1] == 2:
            embedding_ = embedding
        else:
            embedding_ = umap.UMAP(n_neighbors=500).fit_transform(embedding)
        n_points = embedding_.shape[0]
        labels = np.zeros((n_points,))
        labels[np.asarray(subtype_)=="01_TUMOR"] = 0
        labels[np.asarray(subtype_)=="02_STROMA"] = 1
        labels[np.asarray(subtype_)=="03_COMPLEX"] = 2
        labels[np.asarray(subtype_)=="04_LYMPHO"] = 3
        labels[np.asarray(subtype_)=="05_DEBRIS"] = 4
        labels[np.asarray(subtype_)=="06_MUCOSA"] = 5
        labels[np.asarray(subtype_)=="07_ADIPOSE"] = 6
        labels[np.asarray(subtype_)=="08_EMPTY"] = 7
        _, ax = plt.subplots(1, figsize=(14, 10))
        classes = ["TUMOR", "STROMA", "COMPLEX", "LYMPHO", "DEBRIS", "MUCOSA", "ADIPOSE", "EMPTY"]
        n_classes = len(classes)
        plt.scatter(embedding_[:, 0], embedding_[:, 1], s=10, c=labels, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        return plt

    def read_data_into_batches(self, path_dataset):
        img_ext = '.npy'
        paths_of_images = [glob.glob(path_dataset+"\\**\\*"+img_ext)]
        paths_of_images = paths_of_images[0]
        self.n_samples = len(paths_of_images)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        batches = [None] * self.n_batches
        batches_subtypes = [None] * self.n_batches
        for batch_index in range(self.n_batches):
            if batch_index != (self.n_batches-1):
                n_samples_per_batch = self.batch_size
            else:
                n_samples_per_batch = self.n_samples - (self.batch_size * (self.n_batches-1))
            batches[batch_index] = np.zeros((n_samples_per_batch, 128, 128, 3))
            batches_subtypes[batch_index] = [None] * n_samples_per_batch
        for batch_index in range(self.n_batches):
            print("reading batch " + str(batch_index) + "/" + str(self.n_batches-1))
            if batch_index != (self.n_batches-1):
                paths_of_images_of_batch = paths_of_images[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size)]
            else:
                paths_of_images_of_batch = paths_of_images[(batch_index * self.batch_size) :]
            for file_index, filename in enumerate(paths_of_images_of_batch):
                im = np.load(filename)
                batches[batch_index][file_index, :, :, :] = im
                batches_subtypes[batch_index][file_index] = filename.split("\\")[-2]
        return batches, batches_subtypes

    def load_network_model(self, session_):
        # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
        print(" [*] Reading checkpoints...")
        saver = tf.train.Saver()
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir_)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(session_, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            latest_epoch = int(ckpt_name[-1])
            return True, latest_epoch
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # def classification_in_target_domain(self, X, y):
    #     le = preprocessing.LabelEncoder()
    #     le.fit(np.unique(np.asarray(y)))
    #     y = le.transform(y)
    #     clf = LinearSVC(random_state=0, tol=1e-5)
    #     kf = KFold(n_splits=10)
    #     for train_index, test_index in kf.split(X):
    #         X_train, X_test = X[train_index], X[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         clf.fit(X, y)
    #         y_pred = clf.predict(X)

    def classification_in_target_domain(self, X, y, path_save_accuracy_of_test_data, cv=10):
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(np.asarray(y)))
        y = le.transform(y)
        clf = LinearSVC(random_state=0, tol=1e-4, max_iter=10000)
        scores = cross_val_score(clf, X, y, cv=cv)
        if not os.path.exists(path_save_accuracy_of_test_data):
            os.makedirs(path_save_accuracy_of_test_data)
        np.save(path_save_accuracy_of_test_data+"scores.npy", scores)
        np.savetxt(path_save_accuracy_of_test_data+"scores.txt", scores, delimiter=',')  
        str_ = "Mean accuracy: " + str(scores.mean()) + " +- " + str(scores.std())
        print(str_)
        text_file = open(path_save_accuracy_of_test_data+"scores_average.txt", "w")
        text_file.write(str_)
        text_file.close()
        return scores

    def classification_in_target_domain_different_data_portions(self, X, y, path_save_accuracy_of_test_data, proportions, cv=10):
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(np.asarray(y)))
        y = le.transform(y)
        scores_array = np.zeros((len(proportions), cv))
        for proportion_index, proportion in enumerate(proportions):
            print("processing proportion: " + str(proportion) + "....")
            if proportion == 1:
                X_ = X
                y_ = y
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=proportion, random_state=0)
                for train_index, test_index in sss.split(X, y):
                    X_ = X[test_index, :]
                    y_ = y[test_index]
            # scores = cross_val_score(clf, X_, y_, cv=cv)
            skf = StratifiedKFold(n_splits=cv)
            scores = []
            clf = LinearSVC(random_state=0, tol=1e-4, max_iter=10000)
            # clf = RandomForestClassifier(random_state=0)
            for train_index, test_index in skf.split(X_, y_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
                clf.fit(X=X_train, y=y_train)
                scores.append(clf.score(X=X_test, y=y_test))
            del clf
            scores_array[proportion_index, :] = scores
        if not os.path.exists(path_save_accuracy_of_test_data):
            os.makedirs(path_save_accuracy_of_test_data)
        np.save(path_save_accuracy_of_test_data+"scores_array.npy", scores_array)
        np.savetxt(path_save_accuracy_of_test_data+"scores_array.txt", scores_array, delimiter=',')  
        # plot:
        scores_array = scores_array * 100
        proportions = [proportion*100 for proportion in proportions]
        mean_scores = scores_array.mean(axis=1)
        min_scores = scores_array.min(axis=1)
        max_scores = scores_array.max(axis=1)
        plt.fill_between(proportions, min_scores, max_scores, color="r", alpha=0.4)
        plt.plot(proportions, mean_scores, "*-", color="r")
        plt.xlabel("proportion of data (%)")
        plt.ylabel("accuracy (%)")
        plt.ylim(40, 100)
        plt.grid()
        if not os.path.exists(path_save_accuracy_of_test_data):
            os.makedirs(path_save_accuracy_of_test_data)
        plt.savefig(path_save_accuracy_of_test_data + 'plot.png')
        plt.clf()
        plt.close()
        return scores

    

