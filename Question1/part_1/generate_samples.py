import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import yaml

def load_config(config_path="config_training_data.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args =  Config(**load_config())

# Define the Hamiltonian function for various distributions
def functions(coords):
    if args.dist_name == '1D_Gauss_mix':
        # 1D Gaussian Mixture
        q, p = tf.split(coords, num_or_size_splits=2, axis=-1)
        mu1 = 1.0
        mu2 = -1.0
        sigma = 0.35
        term1 = -tf.math.log(
            0.5 * tf.exp(-(q - mu1) ** 2 / (2 * sigma ** 2)) +
            0.5 * tf.exp(-(q - mu2) ** 2 / (2 * sigma ** 2))
        )
        H = term1 + p ** 2 / 2

    elif args.dist_name == '2D_Gauss_mix':
        # 2D Gaussian Four Mixtures
        q1, q2, p1, p2 = tf.split(coords, num_or_size_splits=4, axis=-1)
        sigma_inv = tf.constant([[1., 0.], [0., 1.]])
        term1 = 0.

        mu_list = [
            tf.constant([3., 0.], dtype=tf.float32),
            tf.constant([-3., 0.], dtype=tf.float32),
            tf.constant([0., 3.], dtype=tf.float32),
            tf.constant([0., -3.], dtype=tf.float32)
        ]

        for mu in mu_list:
            q1 = tf.cast(q1, tf.float32)
            q2 = tf.cast(q2, tf.float32)
            y = tf.concat([q1 - mu[0], q2 - mu[1]], axis=-1)
            tmp1 = tf.linalg.matvec(sigma_inv, y)
            term1 += 0.25 * tf.exp(-tf.reduce_sum(y * tmp1, axis=-1, keepdims=True))

        term1 = -tf.math.log(term1)
        term2 = p1 ** 2 / 2 + p2 ** 2 / 2
        H = term1 + term2

    elif args.dist_name == '5D_illconditioned_Gaussian':
        # 5D Ill-Conditioned Gaussian
        dic1 = tf.split(coords, num_or_size_splits=args.input_dim, axis=-1)
        var1 = tf.constant([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02], dtype=tf.float32)
        term1 = dic1[0] ** 2 / (2 * var1[0])
        for ii in range(1, 5):
            term1 += dic1[ii] ** 2 / (2 * var1[ii])
        term2 = 0.0
        for ii in range(5, args.input_dim):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    elif args.dist_name == 'nD_Funnel':
        # nD Funnel Distribution
        dic1 = tf.split(coords, num_or_size_splits=args.input_dim, axis=-1)
        term1 = dic1[0] ** 2 / (2 * 3 ** 2)
        for ii in range(1, int(args.input_dim / 2)):
            term1 += dic1[ii] ** 2 / (2 * tf.exp(dic1[0] / 2) ** 2)
        term2 = 0.0
        for ii in range(int(args.input_dim / 2), args.input_dim):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    elif args.dist_name == 'nD_Rosenbrock':
        # nD Rosenbrock Function
        dic1 = tf.split(coords, num_or_size_splits=args.input_dim, axis=-1)
        term1 = 0.0
        for ii in range(0, int(args.input_dim / 2) - 1):
            term1 += (100.0 * (dic1[ii + 1] - dic1[ii] ** 2) ** 2 + (1 - dic1[ii]) ** 2) / 20.0
        term2 = 0.0
        for ii in range(int(args.input_dim / 2), args.input_dim):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    elif args.dist_name == 'nD_standard_Gaussian':
        # nD Standard Gaussian
        dic1 = tf.split(coords, num_or_size_splits=args.input_dim, axis=-1)
        var1 = tf.ones(args.input_dim, dtype=tf.float32)
        term1 = dic1[0] ** 2 / (2 * var1[0])
        for ii in range(1, int(args.input_dim / 2)):
            term1 += dic1[ii] ** 2 / (2 * var1[ii])
        term2 = 0.0
        for ii in range(int(args.input_dim / 2), args.input_dim):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    else:
        raise ValueError("Probability distribution name not recognized")

    return H

# Define the dynamics function using TensorFlow's automatic differentiation
def dynamics_fn(t, coords):
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)
    coords = tf.reshape(coords, [args.input_dim])

    # 2) Compute Hamiltonian and gradients w.r.t. coords
    with tf.GradientTape() as tape:
        tape.watch(coords)
        H = functions(coords)  
    dcoords = tape.gradient(H, coords)  # shape: (input_dim,)

    # 3) Convert to NumPy for splitting/concatenation approach
    dcoords_np = dcoords.numpy()  # shape (input_dim,)
    dic1 = np.split(dcoords_np, args.input_dim)  # returns list of length input_dim

    # Start S with the first partial wrt p
    half_dim = args.input_dim // 2
    S = np.concatenate([dic1[half_dim]])

    # Append the remaining partials wrt p
    for ii in range(half_dim + 1, args.input_dim):
        S = np.concatenate([S, dic1[ii]])

    # Append negative partials wrt q
    for ii in range(0, half_dim):
        S = np.concatenate([S, -dic1[ii]])

    # 5) Return as a 1D NumPy array
    return S


def leapfrog(dynamics_fn, t_span, y0, n_steps, input_dim):
    """
    dynamics_fn(time, y) -> returns dy/dt (with y splitted into q and p).
    This function must handle splitting y into q,p and computing dq/dt, dp/dt.
    Perform symplectic integration using the leapfrog method.

    Parameters:
    - dynamics_fn: The dynamics function providing time derivatives.
    - t_span: A tuple (start_time, end_time) defining the time horizon.
    - y0: Initial conditions for {q, p}.
    - n_steps: Number of integration steps.
    - input_dim: Dimensionality of the input.
    """
    dt = (t_span[1] - t_span[0]) / n_steps

    dim = len(y0)
    q_dim = dim // 2
    p_dim = dim // 2

    # Store trajectory
    y_traj = [y0.copy()]
    # Evaluate initial derivatives
    aold = dynamics_fn(t_span[0], y0)  # shape: (dim,)

    t = t_span[0]
    y = y0.copy()

    for i in range(1, n_steps + 1):
        # time update
        t_new = t + dt

        # 1) Half-step update for positions
        #    y[:q_dim] = q, y[q_dim:] = p
        for j in range(q_dim):
            y[j] = y[j] + dt * (
                y[q_dim + j] + 0.5 * dt * aold[q_dim + j]
            )

        # Recompute derivatives at new positions
        anew = dynamics_fn(t_new, y)

        # 2) Half-step update for momenta
        for j in range(q_dim):
            y[q_dim + j] = y[q_dim + j] + 0.5 * dt * (
                aold[q_dim + j] + anew[q_dim + j]
            )

        # Add to trajectory
        y_traj.append(y.copy())

        # Update for next iteration
        aold = anew
        t = t_new

    return np.array(y_traj)


# Generate a single trajectory
def get_trajectory(y0=None, **kwargs):
    t_span=[0, args.len_sample]
    timescale = args.len_sample
    n_steps = int(timescale * (t_span[1] - t_span[0]))
    if y0 is None:
        y0 = np.zeros(args.input_dim)
        for ii in range(0, int(args.input_dim // 2)):
            y0[ii] = norm(loc=0, scale=1).rvs()
    lp_ivp = leapfrog(dynamics_fn, t_span, y0, n_steps, args.input_dim)

    # Compute the derivatives
    dydt = np.array([dynamics_fn(None, lp_ivp[i, :]) for i in range(lp_ivp.shape[0])])
    return lp_ivp, dydt

# Utility functions for saving and loading data
def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Generate the dataset
def get_dataset(seed=0, **kwargs):
    samples = args.num_samples
    test_split = (1.0 - args.test_fraction)
    if args.should_load:
        path = '{}/{}.pkl'.format(args.load_dir, args.load_file_name)
        data = from_pickle(path)
        print("Successfully loaded data")
    else:
        data = {'meta': locals()}
        np.random.seed(seed)
        xs, dxs = [], []

        y_init = np.zeros(args.input_dim)
        # Initialize positions to 0, momenta sampled from standard normal distribution
        for ii in range(int(args.input_dim // 2), args.input_dim):
            y_init[ii] = norm(loc=0, scale=1).rvs()

        print('Generating HMC samples for HNN training')

        for s in range(samples):
            print(f'Sample number {s + 1} of {samples}')
            lp_ivp, dydt = get_trajectory(y0=y_init, **kwargs)
            xs.append(lp_ivp)  # shape (n_steps+1, input_dim)
            dxs.append(dydt)
            y_init = np.zeros(args.input_dim)
            final_state = lp_ivp[-1, :]  # last row => final step
            y_init[:args.input_dim // 2] = final_state[:args.input_dim // 2]
            # re-sample momenta
            y_init[args.input_dim // 2:] = norm(loc=0, scale=1).rvs(size=args.input_dim // 2)

        data['coords'] = np.concatenate(xs, axis=0)
        data['dcoords'] = np.concatenate(dxs, axis=0)

        # Make a train/test split
        split_ix = int(len(data['coords']) * test_split)
        split_data = {}
        for k in ['coords', 'dcoords']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
        data = split_data

        # Save data
        path = '{}/{}.pkl'.format(args.save_dir, args.dist_name)
        to_pickle(data, path)

    return data

# Sample script to use the module
if __name__ == "__main__":
    # Generate the dataset
    args =  Config(**load_config())
    data = get_dataset()

    # Access the training and test data
    coords_train = data['coords']
    dcoords_train = data['dcoords']
    coords_test = data['test_coords']
    dcoords_test = data['test_dcoords']

    print(f"Training coords shape: {coords_train.shape}")
    print(f"Training dcoords shape: {dcoords_train.shape}")
    print(f"Test coords shape: {coords_test.shape}")
    print(f"Test dcoords shape: {dcoords_test.shape}")
    print("done")

    plt.scatter(data['coords'][:, 0], data['coords'][:, 1], s=2)
    plt.title("Training coords: q vs p")
    plot_path = "./training_phase_plot.png"
    plt.savefig(plot_path)
