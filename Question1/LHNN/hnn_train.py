import numpy as np
import os
import pickle
import sys
import yaml
import tensorflow as tf
from nn_models import MLP
from hnn import HNN
from utils import L2_loss

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

args =  Config(**load_config())
with open('./'+args.dist_name+'.pkl', 'rb') as f:
    data = pickle.load(f)

# Ensure data structure matches expectations
if 'coords' not in data:
    raise ValueError("Expected 'coords' key in .pkl file")

def train(args):
    # Set random seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Instead of output_dim = args.input_dim, we use args.latent_dim for L-HNN
    nn_model = MLP(input_dim=args.input_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.latent_dim,
                    nonlinearity=args.nonlinearity)
    model = HNN(input_dim=args.input_dim,
                differentiable_model=nn_model,
                field_type=args.field_type)
    model.build(input_shape=(None, args.input_dim))
    optim = tf.keras.optimizers.Adam(learning_rate=args.learn_rate)

    x = tf.Variable(data['coords'], dtype=tf.float32)
    dxdt = tf.constant(data['dcoords'], dtype=tf.float32)
    test_x = tf.Variable(data['test_coords'], dtype=tf.float32)
    test_dxdt = tf.constant(data['test_dcoords'], dtype=tf.float32)

    print('Training L-HNN begins...')
    stats = {'train_loss': [], 'test_loss': []}

    for step in range(args.total_steps + 1):
        # Select a random batch
        ixs = np.random.permutation(x.shape[0])[:args.batch_size]
        x_batch = tf.gather(x, ixs)
        dxdt_batch = tf.gather(dxdt, ixs)

        with tf.GradientTape() as tape:
            # Compute predicted time derivatives using the L-HNN
            dxdt_hat = model.time_derivative(x_batch)
            # Compute loss between predicted and true derivatives
            loss = L2_loss(dxdt_batch, dxdt_hat)

        gradients = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))

        # Evaluate on test data
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        stats['train_loss'].append(loss.numpy())
        stats['test_loss'].append(test_loss.numpy())

        if step % args.print_every == 0:
            print(f"step {step}, train_loss {loss.numpy():.4e}, test_loss {test_loss.numpy():.4e}")

    # Final evaluation
    train_dxdt_hat = model.time_derivative(x)
    train_dist = tf.square(dxdt - train_dxdt_hat)
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = tf.square(test_dxdt - test_dxdt_hat)

    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(tf.reduce_mean(train_dist).numpy(),
                  tf.math.reduce_std(train_dist).numpy() / np.sqrt(train_dist.shape[0]),
                  tf.reduce_mean(test_dist).numpy(),
                  tf.math.reduce_std(test_dist).numpy() / np.sqrt(test_dist.shape[0])))

    return model, stats

model, stats = train(args)
path = f'{args.save_dir}/{args.dist_name}'+ "_inter.weights.h5"
model.save_weights(path)

