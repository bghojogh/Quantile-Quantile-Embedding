# required tensorflow version: 1.14.0

import Utils
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import CNN_Siamese
import ResNet
import numpy as np
import matplotlib.pyplot as plt
# import cv2
from collections import OrderedDict  #--> for not repeating legends in plot
import umap
import os
from Target_domain import Target_domain


def main():
    #================================ settings:
    train_the_source_domain = True
    train_the_target_domain = False
    assert train_the_source_domain != train_the_target_domain
    deep_model = "ResNet"  #--> "CNN", "ResNet"
    n_res_blocks = 18  #--> 18, 34, 50, 101, 152
    batch_size = 32
    learning_rate = 1e-5
    margin_in_loss = 0.25
    feature_space_dimension = 2
    n_classes = 10
    path_save_network_model = ".\\network_model\\" + deep_model + "\\"
    model_dir_ = model_dir(model_name=deep_model, n_res_blocks=n_res_blocks, batch_size=batch_size, learning_rate=learning_rate)
    #================================ 
    if train_the_source_domain:
        train_source_domain(deep_model, n_res_blocks, batch_size, learning_rate, path_save_network_model, model_dir_, feature_space_dimension, margin_in_loss, n_classes)
    if train_the_target_domain:
        batch_size = 500
        train_target_domain(path_save_network_model, model_dir_, deep_model, batch_size, feature_space_dimension, n_res_blocks, margin_in_loss, n_classes)


def train_target_domain(path_save_network_model, model_dir_, deep_model, batch_size, feature_space_dimension, n_res_blocks, margin_in_loss, n_classes):
    #================================ settings:
    path_dataset_of_target_space = "D:\\Datasets\\Kather_train_test\\test"
    path_save_embeddings_of_test_data = ".\\results\\" + deep_model + "\\embedding_test_set\\"
    path_save_accuracy_of_test_data = ".\\results\\" + deep_model + "\\accuracy_test_set\\"
    proportions = [0.04, 0.08, 0.12, 0.16, 0.20, 0.40, 0.60, 0.80, 1]
    #================================ 
    if deep_model == "CNN":
        pass
    elif deep_model == "ResNet":
        network_ = ResNet.ResNet(feature_space_dimension=feature_space_dimension, n_classes=n_classes, n_res_blocks=n_res_blocks, margin_in_loss=margin_in_loss, is_train=False)
    target_domain = Target_domain(path_save_network_model, model_dir_, deep_model, batch_size, feature_space_dimension)
    batches, batches_subtypes = target_domain.read_data_into_batches(path_dataset=path_dataset_of_target_space)
    embedding, subtypes = target_domain.embed_data_in_the_source_domain(batches, batches_subtypes, network_, path_save_embeddings_of_test_data)
    # target_domain.classification_in_target_domain(X=embedding, y=subtypes, path_save_accuracy_of_test_data=path_save_accuracy_of_test_data, cv=10)
    target_domain.classification_in_target_domain_different_data_portions(X=embedding, y=subtypes, path_save_accuracy_of_test_data=path_save_accuracy_of_test_data, 
                                                                          proportions=proportions, cv=10)


def train_source_domain(deep_model, n_res_blocks, batch_size, learning_rate, path_save_network_model, model_dir_, feature_space_dimension, margin_in_loss, n_classes):
    #================================ settings:
    save_plot_embedding_space = True
    save_points_in_embedding_space = True
    load_saved_network_model = False
    save_network_model_every_how_many_epochs = 2
    save_embedding_every_how_many_epochs = 2
    num_epoch = 50
    # STEPS_PER_EPOCH_TRAIN = 704
    STEPS_PER_EPOCH_TRAIN = 312  #--> 10000/32 (n_samples/batch_size)
    n_samples_plot = None   #--> if None, plot all
    # path_tfrecords_train = 'D:\\triplet_work_Main\\_Important_files\\triplets.tfrecords'
    # path_tfrecords_train = '.\\..\\..\\triplet_work_Main\\_Important_files\\triplets.tfrecords'
    # path_tfrecords_train = 'D:\\Datasets\\Kather_train_NotTriplets\\images.tfrecords'
    # path_tfrecords_train = 'C:\\Users\\benya\\Desktop\\my_PhD\\QQE\\codes\\3_deep_codes\\dataset\\MNIST_1000\\data.tfrecords'
    path_tfrecords_train = '.\\data.tfrecords'
    path_save_embedding_space = ".\\results\\" + deep_model + "\\embedding_train_set\\"
    path_save_loss = ".\\loss_saved\\"
    #================================ 

    train_dataset = tf.data.TFRecordDataset([path_tfrecords_train])
    train_dataset = train_dataset.map(Utils.parse_function)
    train_dataset = train_dataset.map(Utils.normalize_images)

    num_repeat = None
    train_dataset = train_dataset.repeat(num_repeat)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(batch_size)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                             train_dataset.output_shapes)

    next_element = iterator.get_next()
    # training_iterator = train_dataset.make_initializable_iterator()
    training_iterator = tf.data.make_initializable_iterator(train_dataset)

    if deep_model == "CNN":
        pass
    elif deep_model == "ResNet":
        network_ = ResNet.ResNet(feature_space_dimension=feature_space_dimension, n_classes=n_classes, n_res_blocks=n_res_blocks, margin_in_loss=margin_in_loss, is_train=True)

    # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(network_.loss)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(network_.loss)
    # tf.initialize_all_variables().run()

    saver_ = tf.train.Saver(max_to_keep=None)  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        training_handle = sess.run(training_iterator.string_handle())
        sess.run(training_iterator.initializer)

        if load_saved_network_model:
            _, latest_epoch = load_network_model(saver_=saver_, session_=sess, checkpoint_dir=path_save_network_model,
                                                                model_dir_=model_dir_, model_name=deep_model)
        else:
            latest_epoch = -1

        loss_average_of_epochs = []
        for epoch in range(latest_epoch+1, num_epoch):
            losses_in_epoch = []
            print("============= epoch: " + str(epoch) + "/" + str(num_epoch-1))
            embeddings_in_epoch = np.zeros((STEPS_PER_EPOCH_TRAIN * batch_size, feature_space_dimension))
            # types_in_epoch = np.zeros((STEPS_PER_EPOCH_TRAIN * batch_size * 3,))
            subtypes_in_epoch = np.zeros((STEPS_PER_EPOCH_TRAIN * batch_size,))
            for i in range(STEPS_PER_EPOCH_TRAIN):
                # image_anchor, image_neighbor, image_distant, type_anchor, type_neighbor, type_distant, subtype_anchor, subtype_neighbor, subtype_distant = sess.run(next_element,
                #                                                        feed_dict={handle: training_handle})

                # image_anchor, image_neighbor, image_distant, subtype_anchor, subtype_neighbor, subtype_distant = sess.run(next_element,
                #                                                        feed_dict={handle: training_handle})

                image_, tissue_type = sess.run(next_element,
                                      feed_dict={handle: training_handle})

                # image_anchor_batch_3_channels = image_anchor.reshape((batch_size, 128, 128, 4))[:,:,:,0:3]
                # image_neighbor_batch_3_channels = image_neighbor.reshape((batch_size, 128, 128, 4))[:,:,:,0:3]
                # image_distant_batch_3_channels = image_distant.reshape((batch_size, 128, 128, 4))[:,:,:,0:3]

                # print(tissue_type.shape)
                # print(image_.shape)
                # print(tissue_type)
                # print(np.unique(tissue_type))
                image_batch_ = image_.reshape((batch_size, 28, 28, 1))
                # plt.imshow(image_batch_[0, :, :, 0].reshape((28, 28)), interpolation='nearest')
                # plt.show()

                _, loss_v, embedding1 = sess.run([train_step, network_.loss, network_.o1], feed_dict={
                    network_.x1: image_batch_,
                    network_.label_: tissue_type})

                embeddings_in_epoch[ ((i*batch_size)+(0*batch_size)) : ((i*batch_size)+(1*batch_size)), : ] = embedding1
                subtypes_in_epoch[ ((i*batch_size)+(0*batch_size)) : ((i*batch_size)+(1*batch_size)) ] = tissue_type

                losses_in_epoch.extend([loss_v])
                
            # report average loss of epoch:
            loss_average_of_epochs.append(np.average(np.asarray(losses_in_epoch)))
            print("Average loss of epoch " + str(epoch) + ": " + str(loss_average_of_epochs[-1]))
            if not os.path.exists(path_save_loss):
                os.makedirs(path_save_loss)
            np.save(path_save_loss + "loss.npy", np.asarray(loss_average_of_epochs))

            # plot the embedding space:
            if (epoch % save_embedding_every_how_many_epochs == 0):
                if save_points_in_embedding_space:
                    if not os.path.exists(path_save_embedding_space+"numpy\\"):
                        os.makedirs(path_save_embedding_space+"numpy\\")
                    np.save(path_save_embedding_space+"numpy\\embeddings_in_epoch_" + str(epoch) + ".npy", embeddings_in_epoch)
                    # np.save(path_save_embedding_space+"numpy\\types_in_epoch_" + str(epoch) + ".npy", types_in_epoch)
                    np.save(path_save_embedding_space+"numpy\\subtypes_in_epoch_" + str(epoch) + ".npy", subtypes_in_epoch)
                if save_plot_embedding_space:
                    print("saving the plot of embedding space....")
                    plt.figure(200)
                    # fig.clf()
                    # TCGA_get_color_and_shape_of_points(embeddings_in_epoch, types_in_epoch, subtypes_in_epoch, n_samples_plot)
                    Kather_get_color_and_shape_of_points(embeddings_in_epoch, subtypes_in_epoch, n_samples_plot)
                    if not os.path.exists(path_save_embedding_space+"plots\\"):
                        os.makedirs(path_save_embedding_space+"plots\\")
                    plt.savefig(path_save_embedding_space+"plots\\" + 'epoch' + str(epoch) + '_step' + str(i) + '.png')
                    plt.clf()
                    plt.close()

            # save the network model:
            if (epoch % save_network_model_every_how_many_epochs == 0):
                save_network_model(saver_=saver_, session_=sess, checkpoint_dir=path_save_network_model, step=epoch, model_name=deep_model, model_dir_=model_dir_)
                print("Model saved in path: %s" % path_save_network_model)
                # network_.save_network(session_=sess, checkpoint_dir=path_save_network_model)

def TCGA_get_color_and_shape_of_points(embedding, type_, subtype_, n_samples_plot=None):
    n_samples = embedding.shape[0]
    if n_samples_plot != None:
        indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
    else:
        indices_to_plot = np.random.choice(range(n_samples), n_samples, replace=False)
    embedding_sampled = embedding[indices_to_plot, :]
    if embedding.shape[1] == 2:
        pass
    else:
        embedding_sampled = umap.UMAP(n_neighbors=500).fit_transform(embedding_sampled)
    n_points = embedding.shape[0]
    # n_points_sampled = embedding_sampled.shape[0]
    labels = np.zeros((n_points,))
    labels[(type_==0) & (subtype_==0)] = 0
    labels[(type_==0) & (subtype_==1)] = 1
    labels[(type_==0) & (subtype_==2)] = 2
    labels[(type_==1) & (subtype_==0)] = 3
    labels[(type_==1) & (subtype_==1)] = 4
    labels[(type_==1) & (subtype_==2)] = 5
    labels[(type_==1) & (subtype_==3)] = 6
    labels[(type_==2) & (subtype_==0)] = 7
    labels[(type_==2) & (subtype_==1)] = 8
    labels_sampled = labels[indices_to_plot]
    _, ax = plt.subplots(1, figsize=(14, 10))
    classes = ["LUSC", "MESO", "LUAD", "READ", "COAD", "STAD", "ESCA", "TGCT", "PRAD"]
    n_classes = len(classes)
    plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels_sampled, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(classes)
    return plt

def Kather_get_color_and_shape_of_points(embedding, subtype_, n_samples_plot=None):
    n_samples = embedding.shape[0]
    if n_samples_plot != None:
        indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
    else:
        indices_to_plot = np.random.choice(range(n_samples), n_samples, replace=False)
    embedding_sampled = embedding[indices_to_plot, :]
    # print(np.isnan(embedding_sampled).any())
    if embedding.shape[1] == 2:
        pass
    else:
        embedding_sampled = umap.UMAP(n_neighbors=500).fit_transform(embedding_sampled)
    n_points = embedding.shape[0]
    # n_points_sampled = embedding_sampled.shape[0]
    labels = np.zeros((n_points,))
    labels[(subtype_==0)] = 0
    labels[(subtype_==1)] = 1
    labels[(subtype_==2)] = 2
    labels[(subtype_==3)] = 3
    labels[(subtype_==4)] = 4
    labels[(subtype_==5)] = 5
    labels[(subtype_==6)] = 6
    labels[(subtype_==7)] = 7
    labels[(subtype_==8)] = 8
    labels[(subtype_==9)] = 9
    labels_sampled = labels[indices_to_plot]
    _, ax = plt.subplots(1, figsize=(14, 10))
    classes = ['0','1','2','3','4','5','6','7','8','9']
    n_classes = len(classes)
    # plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels_sampled, cmap='Spectral', alpha=1.0)
    plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels_sampled, cmap='tab10', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(classes)
    return plt



def save_network_model(saver_, session_, checkpoint_dir, step, model_name, model_dir_):
    # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    # https://github.com/taki0112/ResNet-Tensorflow/blob/master/ResNet.py
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir_)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver_.save(session_, os.path.join(checkpoint_dir, model_name+'.model'), global_step=step)

def load_network_model(saver_, session_, checkpoint_dir, model_dir_, model_name):
    # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir_)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver_.restore(session_, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        latest_epoch = int(ckpt_name[-1])
        return True, latest_epoch
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def model_dir(model_name, n_res_blocks, batch_size, learning_rate):
    return "{}_{}_{}_{}".format(model_name, n_res_blocks, batch_size, learning_rate)


if __name__ == "__main__":
    main()