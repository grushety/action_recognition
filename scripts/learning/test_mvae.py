import numpy as np
import tensorflow as tf
import scipy.io
import math
import sys
import os

from mvae import VariationalAutoencoder
from mvae import network_param

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
PATH = "/home/yulia/pepper_ws/src/action_recognition/scripts/learning"

data = scipy.io.loadmat(PATH + "/database/original_normalized_data.mat")
X_test = 1 * data["data"]
print(X_test.shape)

with tf.Graph().as_default() as g:
    with tf.Session() as sess:
        # Network parameters
        network_architecture = network_param()
        learning_rate = 0.00001
        batch_size = 1

        model = VariationalAutoencoder(sess, network_architecture, batch_size=batch_size, learning_rate=learning_rate,
                                       vae_mode=False, vae_mode_modalities=False)
    with tf.Session() as sess:
        new_saver = tf.train.Saver()
        new_saver.restore(sess, PATH + "/models/all_conf_network.ckpt")
        print("Model restored.")

    # Prediction test

    # Reconstruction test
        print('Test 1')
        for x in X_test:
            #print x
            x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct(sess,  [x])
            print x_reconstruct[0][6:]
            x_pred, x_pred_sigma = model.reconstruct(sess, x_reconstruct)
            print x_pred[0][6:]



