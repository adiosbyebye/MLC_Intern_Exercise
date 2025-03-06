import os
import sys
import pickle
import yaml
import unittest
import numpy as np
import tensorflow as tf
from utils import choose_nonlinearity, L2_loss
from hnn import HNN
from nn_models import MLP


###############################################################################
# Configuration container
###############################################################################
class Config:
    """
    Stores configuration parameters as object attributes.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

###############################################################################
# 1) Training loop
###############################################################################
def train(args, data):
    """
    Sets up the MLP -> HNN model and trains it on provided PM-HMC data.
    """
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.dist_name == "GLMM":
        args.dim_u = args.S * args.N * args.dim_per_subject

    input_dim = 2 * (args.dim_theta + args.dim_u)

    mlp = MLP(input_dim=input_dim,
              hidden_dim=args.hidden_dim,
              output_dim=1,
              nonlinearity=args.nonlinearity)

    model = HNN(input_dim=input_dim,
                mlp_model=mlp,
                d=args.dim_theta,
                D=args.dim_u)
    model.build(input_shape=(None, input_dim))

    optim = tf.keras.optimizers.Adam(learning_rate=args.learn_rate)

    coords      = tf.Variable(data['coords'],    dtype=tf.float32)
    dcoords     = tf.constant(data['dcoords'],   dtype=tf.float32)
    coords_test = tf.Variable(data['test_coords'],  dtype=tf.float32)
    dcoords_test= tf.constant(data['test_dcoords'], dtype=tf.float32)

    stats = {'train_loss': [], 'test_loss': []}

    def get_loss(x_batch, dxdt_batch):
        dxdt_hat = model.time_derivative(x_batch)
        return L2_loss(dxdt_hat, dxdt_batch)

    num_data = coords.shape[0]
    for step in range(args.total_steps + 1):
        ixs = np.random.permutation(num_data)[:args.batch_size]
        x_batch = tf.gather(coords, ixs)
        dxdt_batch = tf.gather(dcoords, ixs)

        with tf.GradientTape() as tape:
            loss = get_loss(x_batch, dxdt_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))

        test_loss = get_loss(coords_test, dcoords_test)
        stats['train_loss'].append(loss.numpy())
        stats['test_loss'].append(test_loss.numpy())

        if step % args.print_every == 0:
            print(f"Step {step}, train_loss={loss.numpy():.4e}, test_loss={test_loss.numpy():.4e}")

    return model, stats


###############################################################################
# 2) Unit tests for code functionality
###############################################################################
class TestHNN(unittest.TestCase):
    def test_mlp_forward_pass(self):
        """
        Tests whether MLP returns an output of expected shape for a given input.
        """
        test_mlp = MLP(input_dim=4, hidden_dim=8, output_dim=1, nonlinearity='relu')
        test_mlp.build(input_shape=(None, 4))
        x = tf.random.normal([5, 4])
        y = test_mlp(x)
        self.assertEqual(y.shape, (5, 1))

    def test_hnn_time_derivative(self):
        """
        Tests whether HNN computes a time derivative of expected shape.
        """
        d, D = 2, 3
        input_dim = 2*(d + D)
        test_mlp = MLP(input_dim=input_dim, hidden_dim=8, output_dim=1, nonlinearity='relu')
        test_hnn = HNN(input_dim=input_dim, mlp_model=test_mlp, d=d, D=D)
        test_hnn.build(input_shape=(None, input_dim))

        x = tf.random.normal([5, input_dim])
        dxdt = test_hnn.time_derivative(x)
        self.assertEqual(dxdt.shape, (5, input_dim))

###############################################################################
# Main script
###############################################################################
if __name__ == '__main__':
    # If an argument "test" is passed, run the unit tests.
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[''], exit=False)
    else:
        # Load configuration
        with open("config.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        args = Config(**config_dict)

        # Load dataset
        pkl_path = os.path.join(args.save_dir, f"{args.dist_name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Error: data file not found at {pkl_path}")
            sys.exit(1)

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Train the model
        model, stats = train(args, data)

        # Save the model weights
        weight_path = os.path.join(args.save_dir, f"{args.dist_name}_pmHMC.weights.h5")
        model.save_weights(weight_path)
        print("Training finished and weights saved to", weight_path)


