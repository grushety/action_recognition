import numpy as np
import tensorflow as tf
import scipy.io
import math
import sys
import os

#last results
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
PATH = "/home/yulia/pepper_ws/src/action_recognition/scripts/learning"

# load data set
data = scipy.io.loadmat(PATH + "/database/augm_rec_pred_data.mat")
X_init = 1 * data["data"]
print(X_init.shape)
n_samples = X_init.shape[0]



class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian distributions and
    realized by multi-layer perceptrons. The VAE can be learned end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, sess, network_architecture, transfer_fct=tf.nn.relu, learning_rate=0.001, batch_size=100,
                 vae_mode=False, vae_mode_modalities=False):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate

        #The batch size defines the number of samples that will be propagated through the network
        self.batch_size = batch_size
        #vae_mode the way to define self.z
        self.vae_mode = vae_mode
        #
        self.vae_mode_modalities = vae_mode_modalities
        #self.n_mc = 4
        #self.n_vis = 4

        self.n_input = network_architecture['n_input']
        self.n_z = network_architecture['n_z']

        # tf Graph input
        self.x = tf.placeholder(tf.float64, [None, self.n_input], name='InputData')
        self.x_noiseless = tf.placeholder(tf.float64, [None, self.n_input], name='NoiselessData')

        self.layers = {}

        self.n_epoch = tf.zeros([], tf.float64)

        # Create autoencoder network
        self._create_network()

        # Define loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()  # tf.initialize_all_variables() #

        # Launch the session
        self.sess = sess  # tf.InteractiveSession()
        self.sess.run(init)

        self.saver = tf.train.Saver()

        # Summary monitors
        tf.summary.scalar("loss", self.cost)  # tf.summary.FileWriter(logs_path) #
        # tf.summary.scalar("loss_J",self.cost_J)
        self.merged_summary_op = tf.summary.merge_all()  # tf.merge_all_summaries()

    def _slice_input(self, input_layer, size_mod):
        slices = []
        count = 0
        for i in range(len(self.network_architecture[size_mod])):
            new_slice = tf.slice(input_layer, [0, count], [self.batch_size, self.network_architecture[size_mod][
                i]])  # tf.slice(layer_2, [0,200], [105,100])
            count += self.network_architecture[size_mod][i]
            slices.append(new_slice)
        return slices

    def _create_partial_network(self, name, input_layer):
        with tf.name_scope(name):
            self.layers[name] = [input_layer]
            for i in range(len(self.network_architecture[name])):
                h = tf.Variable(
                    xavier_init(int(self.layers[name][-1].get_shape()[1]), self.network_architecture[name][i]))
                b = tf.Variable(tf.zeros([self.network_architecture[name][i]], dtype=tf.float64))
                layer = self.transfer_fct(tf.add(tf.matmul(self.layers[name][-1], h), b))
                self.layers[name].append(layer)

    def _create_variational_network(self, input_layer, latent_size):
        input_layer_size = int(input_layer.get_shape()[1])

        h_mean = tf.Variable(xavier_init(input_layer_size, latent_size))
        h_var = tf.Variable(xavier_init(input_layer_size, latent_size))
        b_mean = tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        b_var = tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        mean = tf.add(tf.matmul(input_layer, h_mean), b_mean)
        log_sigma_sq = tf.log(tf.exp(tf.add(tf.matmul(input_layer, h_var), b_var)) + 0.0001)
        return mean, log_sigma_sq

    #read it
    def _create_modalities_network(self, names, slices):
        for i in range(len(names)):
            self._create_partial_network(names[i], slices[i])

    def _create_mod_variational_network(self, names, sizes_mod):
        assert len(self.network_architecture[sizes_mod]) == len(names)
        sizes = self.network_architecture[sizes_mod]
        self.layers['final_means'] = []
        self.layers['final_sigmas'] = []
        for i in range(len(names)):
            mean, log_sigma_sq = self._create_variational_network(self.layers[names[i]][-1], sizes[i])
            self.layers['final_means'].append(mean)
            self.layers['final_sigmas'].append(log_sigma_sq)
        global_mean = tf.concat(self.layers['final_means'], 1)
        global_sigma = tf.concat(self.layers['final_sigmas'], 1)
        self.layers["global_mean_reconstr"] = [global_mean]
        self.layers["global_sigma_reconstr"] = [global_sigma]
        return global_mean, global_sigma

    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.x_noiseless_sliced = self._slice_input(self.x_noiseless, 'size_slices')
        slices = self._slice_input(self.x, 'size_slices')
        self._create_modalities_network(['mod0', 'mod1'], slices)

        self.output_mod = tf.concat([self.layers['mod0'][-1], self.layers['mod1'][-1]], 1)
        self.layers['concat'] = [self.output_mod]

        # self._create_partial_network('enc_shared',self.x)
        self._create_partial_network('enc_shared', self.output_mod)
        self.z_mean, self.z_log_sigma_sq = self._create_variational_network(self.layers['enc_shared'][-1], self.n_z)

        if self.vae_mode:
            eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float64)
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        else:
            self.z = self.z_mean

        self._create_partial_network('dec_shared', self.z)

        slices_shared = self._slice_input(self.layers['dec_shared'][-1], 'size_slices_shared')
        self._create_modalities_network(['mod0_2', 'mod1_2'], slices_shared)

        self.x_reconstr, self.x_log_sigma_sq = self._create_mod_variational_network(
            ['mod0_2', 'mod1_2'], 'size_slices')

        # self.output_mod_shared = tf.concat([self.layers['mod0_2'][-1],self.layers['mod1_2'][-1],self.layers['mod2_2'][-1],self.layers['mod3_2'][-1],self.layers['mod4_2'][-1]],1)
        # self.x_reconstr, self.x_log_sigma_sq = self._create_variational_network(self.output_mod_shared,self.n_input)

    def _create_loss_optimizer(self):
        with tf.name_scope('Loss_Opt'):
            self.alpha = 1 - tf.minimum(self.n_epoch / 1000, 1)

            self.tmp_costs = []
            # 1.) The reconstruction loss (the negative log probability
            #     of the input under the reconstructed Bernoulli distribution
            #     induced by the decoder in the data space).
            #     This can be interpreted as the number of "nats" required
            #     for reconstructing the input when the activation in latent
            #     is given.
            # Adding 1e-10 to avoid evaluation of log(0.0)
            for i in range(len(self.layers['final_means'])): #10
                #reduce sum added all digit in column
                reconstr_loss = (0.5 * tf.reduce_sum( #quadrat of dif btw orig and gen data
                    tf.square(self.x_noiseless_sliced[i] - self.layers['final_means'][i]) / tf.exp(
                        self.layers['final_sigmas'][i]), 1)   # e^x could be only positive
                                 + 0.5 * abs(tf.reduce_sum(self.layers['final_sigmas'][i], 1)) # only thing can be negative
                                 + 0.5 * self.n_z / 2 * np.log(2 * math.pi)) / self.network_architecture['size_slices'][
                                    i]
                self.tmp_costs.append(reconstr_loss)
            self.reconstr_loss = tf.reduce_mean(
                self.tmp_costs[0] + self.tmp_costs[1])

            # 2.) The latent loss, which is defined as the Kullback Leibler divergence
            ##    between the distribution in latent space induced by the encoder on
            #     the data and some prior. This acts as a kind of regularizer.
            #     This can be interpreted as the number of "nats" required
            #     for transmitting the the latent space distribution given
            #     the prior.
            self.latent_loss = -0.5 * tf.reduce_sum(
                1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

            self.cost = tf.reduce_mean(self.reconstr_loss + tf.scalar_mul(self.alpha, self.latent_loss))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

            self.m_reconstr_loss = self.reconstr_loss
            self.m_latent_loss = tf.reduce_mean(self.latent_loss)

    def partial_fit(self, sess, X, X_noiseless, epoch):
        """Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        opt, cost, recon, latent, x_rec, alpha = sess.run(
            (self.optimizer, self.cost, self.m_reconstr_loss, self.m_latent_loss, self.x_reconstr, self.alpha),
            feed_dict={self.x: X, self.x_noiseless: X_noiseless, self.n_epoch: epoch})
        return cost, recon, latent, x_rec, alpha

    def transform(self, sess, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        return sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, sess, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is generated. Otherwise, z_mu is drawn from prior in latent space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.n_z)
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        return sess.run(self.x_reconstr, feed_dict={self.z: z_mu})

    def reconstruct(self, sess, X_test):
        """ Use VAE to reconstruct given data. """
        x_rec_mean, x_rec_log_sigma_sq = sess.run((self.x_reconstr, self.x_log_sigma_sq),
                                                  feed_dict={self.x: X_test})
        return x_rec_mean, x_rec_log_sigma_sq

    def print_layers_size(self):
        print(self.cost)
        for layer in self.layers:
            print(layer)
            for l in self.layers[layer]:
                print(l)

#########################################################################################################

def train(sess, vae, input_data, learning_rate=0.0001, batch_size=100, training_epochs=10, display_step=1,
                vae_mode=True, vae_mode_modalities=True):
    # Write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(PATH + "/logs", graph=tf.get_default_graph())

    # Training cycle for whole network
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_recon = 0.
        avg_latent = 0.
        total_batch = int(n_samples / batch_size)

        X_shuffled = shuffle_data(input_data)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs_augmented = X_shuffled[batch_size * i:batch_size * i + batch_size]

            batch_xs = np.asarray(
                [item[10:] for item in batch_xs_augmented])  # np.asarray([item[:18]   for item in batch_xs_augmented])
            batch_xs_noiseless = np.asarray(
                [item[:10] for item in batch_xs_augmented])  # np.asarray([item[:18]   for item in batch_xs_augmented])
            # batch_xs_noiseless_J   = np.asarray([item[8:12]   for item in batch_xs_noiseless])

            # Fit training using batch data
            cost, recon, latent, x_rec, alpha = vae.partial_fit(sess, batch_xs, batch_xs_noiseless, epoch)
            avg_cost += cost / n_samples * batch_size
            avg_recon += recon / n_samples * batch_size
            avg_latent += latent / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: %04d / %04d, Cost= %04f, Recon= %04f, Latent= %04f, alpha= %04f" % \
                  (epoch, training_epochs, avg_cost, avg_recon, avg_latent, alpha))

    save_path = vae.saver.save(vae.sess, PATH + "/models/test_network.ckpt")


def shuffle_data(x):
    """
    @param x: Input set of samples
    @return: Set of samples in random order
    """
    y = list(x)
    np.random.shuffle(y)
    return np.asarray(y)


def network_param():
    """
    @return: Architecture for 2 modalities MVAE
    """
    network_architecture = {
         'n_input': 10,
         'n_z': 10, # output size
         'size_slices': [6, 4],
         'size_slices_shared': [15, 10],
         'mod0': [30, 15],
         'mod1': [20, 10],
         'mod0_2': [30, 6],
         'mod1_2': [20, 4],
         'enc_shared': [50],
         'dec_shared': [50, 25]}
    return network_architecture


def xavier_init(fan_in, fan_out, constant=1):
    """
    Xavier/Glorot initialization of network weights depends the number of
    incoming connections (fan_in),
    outgoing connections (fan_out),
    and kind of activation function (sigmoid or tanh) of the neuron.
    """
    low = -constant * np.sqrt(1.0 / (fan_in + fan_out))
    high = constant * np.sqrt(1.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float64)
    # stddev = np.sqrt(1.0 / (fan_in + fan_out))
    # return tf.random_normal((fan_in, fan_out), mean = 0.0, stddev=stddev, dtype=tf.float64)


##############################################################################################


if __name__ == '__main__':
    learning_rate = 0.00005
    batch_size = 1000

    # Train Network
    print('Train net')

    sess = tf.InteractiveSession()

    vae_mode = True
    vae_mode_modalities = False

    reload_modalities = False
    reload_shared = False

    vae = VariationalAutoencoder(sess, network_param(), learning_rate=learning_rate, batch_size=batch_size,
                                 vae_mode=vae_mode, vae_mode_modalities=vae_mode_modalities)
    vae.print_layers_size()
    train(sess, vae, X_init, training_epochs=1000, batch_size=batch_size)