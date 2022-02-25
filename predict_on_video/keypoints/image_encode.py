import tensorflow as tf
import numpy as np
import os.path as osp

class ImageEncoder(object):

    def __init__(self, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        cur_dir = osp.dirname(__file__)
        checkpoint_filename = osp.join(cur_dir,"networks","mars-small128.pb")
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x):
        feed_dict = {self.input_var:data_x}
        out = self.session.run(self.output_var, feed_dict=feed_dict)
        return out