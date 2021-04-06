from utils import *
import matplotlib.pyplot as plt
# import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from PIL import Image


path_save = "./results/face_glasses_separation/"

if not os.path.exists(path_save):
    os.makedirs(path_save)

# color_map =  matplotlib.colors.hsv_to_rgb(plt.cm.hsv) # plt.cm.bwr  #--> plt.cm.brg, plt.cm.hsv
# color_map = plt.cm.bwr

def read_dataset():
    path_dataset = "C:/Users/benya/Desktop/my_PhD/QQE/codes/2_QQE/datasets/ORL_glasses/"
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
    y = labels.ravel()
    return X, y

def load_image(address_image, image_height, image_width, do_resize=False, scale=1):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    if do_resize:
        size = int(image_height * scale), int(image_width * scale)
        # img.thumbnail(size, Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = resize(img_arr, (int(img_arr.shape[0]*scale), int(img_arr.shape[1]*scale)), order=5, preserve_range=True, mode="constant")
    return img_arr


X, y = read_dataset()

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
