import os.path
import tensorflow as tf
import project_tests as tests


def load_and_view_model():
    model_filename = './data/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_filename, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    # with tf.Session() as sess:
    #     #model_filename = 'PATH_TO_PB.pb'
    #     with gfile.FastGFile(model_filename, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         g_in = tf.import_graph_def(graph_def)

    train_writer = tf.summary.FileWriter('./log')
    train_writer.add_graph(tf.Session(graph=graph).graph)
    train_writer.flush()
    train_writer.close()


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def validate_load_vgg(vgg_path):
    with tf.Session() as sess:
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

    train_writer = tf.summary.FileWriter('./log', graph=sess.graph)
    train_writer.flush()
    train_writer.close()


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                kernel_regularizer=regularizer)
    output = tf.layers.conv2d_transpose(conv_1x1, vgg_layer4_out.shape[-1], 4, strides=(2, 2), padding='same',
                                         kernel_regularizer=regularizer)
    output = tf.add(output, pool4_out_scaled)

    output = tf.layers.conv2d_transpose(output, vgg_layer3_out.shape[-1], 4, strides=(2, 2), padding='same',
                                        kernel_regularizer=regularizer)

    output = tf.add(output, pool3_out_scaled)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=(8, 8), padding='same',
                                        kernel_regularizer=regularizer)
    return output


def validate_layers(vgg_path, num_classes):
    with tf.Session() as sess:
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        output = layers(layer3_out, layer4_out, layer7_out, num_classes)


if __name__ == "__main__":
    data_dir = './data'
    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    num_classes = 2
    # load_and_view_model()
    # validate_load_vgg(vgg_path)
    #validate_layers(vgg_path, num_classes)
    tests.test_layers(layers)