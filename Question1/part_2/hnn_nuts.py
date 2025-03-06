import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.python.ops.numpy_ops.np_config as np_config
import os
import yaml
import pandas as pd
import unittest
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np_config.enable_numpy_behavior()

# Local imports
from generate_samples import pm_leapfrog_integration, pm_functions, pm_dynamics_fn

random.seed(3)

###############################################################################
# 1) Configuration
###############################################################################
class Config:
    """
    Configuration container that stores parameters as attributes.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_config(config_path="config.yaml"):
    """
    Loads configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

###############################################################################
# 2) Main script logic
###############################################################################
def main():
    """
    Main procedure that loads config, sets up the HNN-based NUTS sampler,
    and saves output samples and a diagnostic histogram.
    """

    # Parse current directory and config
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    args = Config(**load_config())

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Determine input dimension
    input_dim = 2 * (args.dim_theta + args.dim_u)

    # Set HMC / NUTS parameters
    N = args.num_samples
    burn = args.burn
    epsilon = args.len_sample / args.n_steps
    N_lf = args.N_lf
    hnn_threshold = args.hnn_threshold
    lf_threshold = args.lf_threshold

    # Load the trained HNN model
    hnn_model = get_model(args, input_dim)

    # Run NUTS with HNN integrator
    D = input_dim // 2
    M = N
    Madapt = 0
    samples = np.empty((M + Madapt, D), dtype=float)
    momentum_samples = np.empty((M + Madapt, D), dtype=float)

    # Initialize
    theta0 = np.zeros(D, dtype=np.float32)
    y0 = np.random.randn(input_dim).astype(np.float32)
    y0[:D] = theta0
    samples[0, :] = theta0
    momentum_samples[0, :] = y0[D:input_dim]

    HNN_accept = np.ones(M)
    traj_len = np.zeros(M)
    alpha_req = np.zeros(M)
    H_store = np.zeros(M)
    monitor_err = np.zeros(M)
    call_lf = 0
    counter_lf = 0
    is_lf = np.zeros(M)

    print("Beginning NUTS sampling with HNN-based integrator...")

    for m in range(1, M + Madapt):
        print(f"Sample iteration: {m}")
        # Refresh momentum
        y0[D:input_dim] = np.random.randn(D).astype(np.float32)
        y0[:D] = samples[m - 1, :]

        joint_tf = pm_functions(tf.convert_to_tensor(y0[None, :], dtype=tf.float32))
        joint0_val = float(joint_tf.numpy())
        u_val = np.random.uniform(0, np.exp(-joint0_val))
        logu_val = np.log(u_val)

        samples[m, :] = samples[m - 1, :]
        momentum_samples[m, :] = momentum_samples[m - 1, :]

        thetaminus = samples[m - 1, :].copy()
        thetaplus = samples[m - 1, :].copy()
        rminus = y0[D:input_dim].copy()
        rplus = y0[D:input_dim].copy()

        j = 0
        n = 1
        s = 1

        if call_lf:
            counter_lf += 1
        if counter_lf == N_lf:
            call_lf = 0
            counter_lf = 0

        while s == 1 and j < args.max_tree_length:
            v = 1 if (np.random.rand() < 0.5) else -1
            if v == -1:
                (thetaminus, rminus, _thetaplus, _rplus,
                 thetaprime, rprime, nprime, sprime,
                 alpha, nalpha, monitor_val, call_lf) = build_tree(
                    thetaminus, rminus, logu_val, v, j, epsilon, joint0_val, call_lf,
                    hnn_model, input_dim, hnn_threshold, N_lf
                )
            else:
                (_thetaminus, _rminus, thetaplus, rplus,
                 thetaprime, rprime, nprime, sprime,
                 alpha, nalpha, monitor_val, call_lf) = build_tree(
                    thetaplus, rplus, logu_val, v, j, epsilon, joint0_val, call_lf,
                    hnn_model, input_dim, hnn_threshold, N_lf
                )

            if (sprime == 1) and (np.random.rand() < float(nprime) / float(n)):
                samples[m, :] = thetaprime.copy()
                momentum_samples[m, :] = rprime.copy()
                r_sto = rprime.copy()
            n += nprime
            s = sprime and not stop_criterion(thetaminus, thetaplus, rminus, rplus)
            j += 1
            monitor_err[m] = monitor_val

        is_lf[m] = int(call_lf)
        traj_len[m] = j
        alpha_req[m] = alpha

        final_y = np.concatenate([samples[m, :], r_sto], axis=0).reshape(1, -1)
        final_tf = pm_functions(tf.convert_to_tensor(final_y, dtype=tf.float32))
        H_store[m] = float(final_tf.numpy())

    print("Sampling complete!")

    # Save final samples
    final_samples = samples[burn:, :]
    csv_filename = "hnn_nuts_samples.csv"
    save_samples_to_csv(final_samples, csv_filename)

    # Plot histogram of first dimension
    plt.figure(figsize=(7, 5))
    plt.hist(final_samples[:, 0], bins=50, color='skyblue', edgecolor='k')
    plt.xlabel('theta[0]')
    plt.ylabel('Counts')
    plt.title('Histogram of theta[0] (HNN+NUTS post-burn)')
    plt.grid(True, alpha=0.3)
    plt.savefig("hist_theta0_hnn_nuts.png", dpi=300)
    plt.close()

    print("All done. Results are in hnn_nuts_samples.csv and hist_theta0_hnn_nuts.png.")


###############################################################################
# 3) Helper functions
###############################################################################
def get_model(args, input_dim):
    """
    Loads the trained HNN from disk:
      - MLP is built with input_dim -> 1 scalar (Hamiltonian).
      - Weights are loaded from '{dist_name}_pmHMC.weights.h5'.
    """
    from nn_models import MLP
    from hnn import HNN

    mlp_model = MLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        nonlinearity=args.nonlinearity
    )
    hnn_model = HNN(
        input_dim=input_dim,
        mlp_model=mlp_model,
        d=args.dim_theta,
        D=args.dim_u
    )
    hnn_model.build(input_shape=(None, input_dim))

    weight_path = os.path.join(args.save_dir, f"{args.dist_name}_pmHMC.weights.h5")
    hnn_model.load_weights(weight_path)
    print(f"Loaded HNN weights from: {weight_path}")
    return hnn_model

def integrate_model(model, t_span, y0, n):
    """
    Integrates from y0 using the HNN's learned vector field for 'n' steps
    via leapfrog.
    """
    def hnn_deriv(t, np_x):
        x_tf = tf.convert_to_tensor(np_x.reshape(1, -1), dtype=tf.float32)
        dx_tf = model.time_derivative(x_tf)
        return dx_tf.numpy().reshape(-1)

    traj = pm_leapfrog_integration(hnn_deriv, t_span, y0, n_steps=n)
    return traj

def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """
    Stops NUTS tree building if (thetaplus - thetaminus) · rminus < 0
    or (thetaplus - thetaminus) · rplus < 0.
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus) < 0.0) or (np.dot(dtheta, rplus) < 0.0)

def build_tree(theta, r, logu, v, j, epsilon, joint0, call_lf,
               hnn_model, input_dim, hnn_threshold, N_lf, max_tree_depth=30):
    """
    Recursively builds the tree for NUTS using the HNN-based integrator,
    with fallback to numeric leapfrog if needed.
    """
    theta = np.asarray(theta, dtype=np.float32)
    r     = np.asarray(r, dtype=np.float32)
    logu_val = float(logu)
    joint0_val = float(joint0)

    if j > max_tree_depth:
        print("Reached max tree depth -> returning stubs")
        return (theta, r, theta, r, theta, r, 0, 0, 0., 0, 0., call_lf)

    # Base case
    if j == 0:
        t_span = [0, epsilon]
        y1 = np.concatenate((theta, r), axis=0)
        # Reverse momentum if v == -1
        if v == -1:
            y1[input_dim//2:] = -y1[input_dim//2:]

        # Integrate using HNN
        hnn_ivp1 = integrate_model(hnn_model, t_span, y1, n=1)
        yfinal = hnn_ivp1[-1, :]
        # Re-reverse if needed
        if v == -1:
            yfinal[input_dim//2:] = -yfinal[input_dim//2:]

        thetaprime = yfinal[:input_dim//2]
        rprime     = yfinal[input_dim//2:]
        new_state = yfinal.reshape(1, -1)
        joint_tf  = pm_functions(tf.convert_to_tensor(new_state, dtype=tf.float32))
        joint_val = float(joint_tf.numpy())
        monitor_val = logu_val + joint_val

        call_lf_bool = call_lf or (monitor_val > hnn_threshold)
        if call_lf_bool:
            print("Fallback to pure numeric leapfrog!")
            lf_ivp = pm_leapfrog_integration(pm_dynamics_fn, t_span, y1, n_steps=1)
            yfinal_lf = lf_ivp[-1, :]
            if v == -1:
                yfinal_lf[input_dim//2:] = -yfinal_lf[input_dim//2:]
            thetaprime = yfinal_lf[:input_dim//2]
            rprime     = yfinal_lf[input_dim//2:]
            new_state_lf = yfinal_lf.reshape(1, -1)
            joint_tf2 = pm_functions(tf.convert_to_tensor(new_state_lf, dtype=tf.float32))
            joint_val = float(joint_tf2.numpy())
            monitor_val = logu_val + joint_val

        nprime = int(logu_val <= -(joint_val - joint0_val))
        thetaminus = thetaprime.copy()
        thetaplus  = thetaprime.copy()
        rminus     = rprime.copy()
        rplus      = rprime.copy()
        alphaprime   = min(1., np.exp(joint0_val - joint_val))
        nalphaprime  = 1
        sprime       = int(monitor_val <= hnn_threshold)
        call_lf      = call_lf_bool
        return (thetaminus, rminus, thetaplus, rplus,
                thetaprime, rprime, nprime, sprime,
                alphaprime, nalphaprime, monitor_val, call_lf)

    else:
        # Build left subtree
        (thetaminus, rminus, thetaplus, rplus,
         thetaprime, rprime, nprime, sprime,
         alphaprime, nalphaprime, monitor_val, call_lf) = build_tree(
            theta, r, logu_val, v, j - 1, epsilon, joint0_val, call_lf,
            hnn_model, input_dim, hnn_threshold, N_lf
        )

        if sprime == 1:
            # Build right subtree
            if v == -1:
                (thetaminus, rminus, _thetaplus, _rplus,
                 thetaprime2, rprime2, nprime2, sprime2,
                 alpha2, nalpha2, monitor2, call_lf
                ) = build_tree(thetaminus, rminus, logu_val, v, j - 1,
                               epsilon, joint0_val, call_lf,
                               hnn_model, input_dim, hnn_threshold, N_lf)
            else:
                (_thetaminus, _rminus, thetaplus, rplus,
                 thetaprime2, rprime2, nprime2, sprime2,
                 alpha2, nalpha2, monitor2, call_lf
                ) = build_tree(thetaplus, rplus, logu_val, v, j - 1,
                               epsilon, joint0_val, call_lf,
                               hnn_model, input_dim, hnn_threshold, N_lf)
            if (sprime2 == 1) and (nprime + nprime2 > 0):
                rand_val = np.random.uniform()
                if rand_val < float(nprime2) / float(nprime + nprime2):
                    thetaprime = thetaprime2.copy()
                    rprime = rprime2.copy()
            nprime += nprime2
            sprime = int(
                sprime and sprime2
                and not stop_criterion(thetaminus, thetaplus, rminus, rplus)
            )
            alphaprime += alpha2
            nalphaprime += nalpha2
            monitor_val = monitor2
        return (thetaminus, rminus, thetaplus, rplus,
                thetaprime, rprime, nprime, sprime,
                alphaprime, nalphaprime, monitor_val, call_lf)

def save_samples_to_csv(samples, filename):
    """
    Saves multi-dimensional samples to a CSV file.
    """
    import pandas as pd
    df = pd.DataFrame(samples, columns=[f'Dimension_{i+1}' for i in range(samples.shape[1])])
    df.to_csv(filename, index=False)
    print(f"Samples saved to {filename}")

###############################################################################
# 4) Unit Tests
###############################################################################
class TestHNNNUTSSampler(unittest.TestCase):
    def test_stop_criterion(self):
        """
        Tests the stop criterion function to ensure it returns expected booleans.
        """
        thetaminus = np.array([0.0, 0.0])
        thetaplus  = np.array([1.0, 1.0])
        rminus     = np.array([ 0.5,  0.5])
        rplus      = np.array([-0.5, -0.5])
        # dot(dtheta, rminus) = dot([1,1],[0.5,0.5])=1, dot(dtheta, rplus)= -1
        # Should return True if either dot < 0
        self.assertTrue(stop_criterion(thetaminus, thetaplus, rminus, rplus))

###############################################################################
# Single main script entry
###############################################################################
if __name__ == "__main__":
    # If an argument "test" is provided, run unit tests
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[''], exit=False)
    else:
        main()
