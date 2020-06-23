
import tensorflow as tf




# def normalize_triplets(image_anchor, image_neighbor, image_distant, type_anchor, type_neighbor, type_distant, subtype_anchor, subtype_neighbor, subtype_distant):

#     image_anchor = tf.cast(image_anchor, tf.float32) * (1. / 255) - 0.5
#     image_neighbor = tf.cast(image_neighbor, tf.float32) * (1. / 255) - 0.5
#     image_distant = tf.cast(image_distant, tf.float32) * (1. / 255) - 0.5   
#     return image_anchor, image_neighbor, image_distant, type_anchor, type_neighbor, type_distant, subtype_anchor, subtype_neighbor, subtype_distant

# def normalize_triplets(image_anchor, image_neighbor, image_distant, subtype_anchor, subtype_neighbor, subtype_distant):

#     image_anchor = tf.cast(image_anchor, tf.float32) * (1. / 255) - 0.5
#     image_neighbor = tf.cast(image_neighbor, tf.float32) * (1. / 255) - 0.5
#     image_distant = tf.cast(image_distant, tf.float32) * (1. / 255) - 0.5   
#     return image_anchor, image_neighbor, image_distant, subtype_anchor, subtype_neighbor, subtype_distant

def normalize_images(image_, subtype):
    image_ = tf.cast(image_, tf.float32) * (1. / 255) - 0.5 
    return image_, subtype

def parse_function(serialized):
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    # features = \
    #     {
    #         'image_anchor': tf.io.FixedLenFeature([], tf.string),
    #         'image_neighbor': tf.io.FixedLenFeature([], tf.string),
    #         'image_distant': tf.io.FixedLenFeature([], tf.string),
    #         'type_anchor': tf.io.FixedLenFeature([], tf.int64),
    #         'type_neighbor': tf.io.FixedLenFeature([], tf.int64),
    #         'type_distant': tf.io.FixedLenFeature([], tf.int64),
    #         'subtype_anchor': tf.io.FixedLenFeature([], tf.int64),
    #         'subtype_neighbor': tf.io.FixedLenFeature([], tf.int64),
    #         'subtype_distant': tf.io.FixedLenFeature([], tf.int64)
    #     }

    # features = \
    #     {
    #         'image_anchor': tf.io.FixedLenFeature([], tf.string),
    #         'image_neighbor': tf.io.FixedLenFeature([], tf.string),
    #         'image_distant': tf.io.FixedLenFeature([], tf.string),
    #         'tissue_type_anchor': tf.io.FixedLenFeature([], tf.int64),
    #         'tissue_type_neighbor': tf.io.FixedLenFeature([], tf.int64),
    #         'tissue_type_distant': tf.io.FixedLenFeature([], tf.int64)
    #     }

    features = \
    {
        'X': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_example = tf.io.parse_single_example(serialized=serialized,
                                             features=features)

    # image_anchor = tf.decode_raw(parsed_example['image_anchor'], tf.uint8)
    # image_neighbor = tf.decode_raw(parsed_example['image_neighbor'], tf.uint8)
    # image_distant = tf.decode_raw(parsed_example['image_distant'], tf.uint8)

    # https://www.tensorflow.org/api_docs/python
    image_ = tf.compat.v1.decode_raw(parsed_example['X'], tf.uint8)
    # image_neighbor = tf.compat.v1.decode_raw(parsed_example['image_neighbor'], tf.uint8)
    # image_distant = tf.compat.v1.decode_raw(parsed_example['image_distant'], tf.uint8)
    # type_anchor = parsed_example['type_anchor']
    # type_neighbor = parsed_example['type_neighbor']
    # type_distant = parsed_example['type_distant']
    # subtype_anchor = parsed_example['subtype_anchor']
    # subtype_neighbor = parsed_example['subtype_neighbor']
    # subtype_distant = parsed_example['subtype_distant']
    tissue_type = parsed_example['label']
    # subtype_neighbor = parsed_example['tissue_type_neighbor']
    # subtype_distant = parsed_example['tissue_type_distant']

    image_.set_shape((IMAGE_PIXELS))
    # image_neighbor.set_shape((IMAGE_PIXELS))
    # image_distant.set_shape((IMAGE_PIXELS))

    # return image_anchor, image_neighbor, image_distant, type_anchor, type_neighbor, type_distant, subtype_anchor, subtype_neighbor, subtype_distant
    return image_, tissue_type
    
def Global_Average_Pooling(x, stride=1) :
        width = np.shape(x)[1]
        height = np.shape(x)[2]
        pool_size = [width, height]
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) 
        # The stride value does not matter