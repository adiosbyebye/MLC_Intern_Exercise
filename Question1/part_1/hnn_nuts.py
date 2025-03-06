import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()
import os
import yaml
import unittest
from tqdm import tqdm
from nn_models import MLP
from hnn import HNN
from generate_samples import leapfrog
from generate_samples import functions, dynamics_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

args =  Config(**load_config())


N = args.num_samples   # number of samples
burn = args.burn # number of burn-in samples
epsilon = args.step_size # step size
N_lf = args.N_lf  # "cool-down" steps if integration errors are high
hnn_threshold = args.hnn_threshold   # HNN integration error threshold
lf_threshold = args.lf_threshold  # Numerical gradient integration error threshold

def get_model(args, baseline):
    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                 field_type=args.field_type, baseline=baseline)
    model.build(input_shape=(None, args.input_dim))
    path = args.dist_name + "_inter.weights.h5"
    model.load_weights(path)
    return model

hnn_model = get_model(args, baseline=False)

def integrate_model(model, t_span, y0, n, **kwargs):
    """Integrate the model dynamics using HNN or leapfrog as needed."""
    def fun(t, np_x):
        x_tf = tf.convert_to_tensor(np_x.reshape(1, args.input_dim), dtype=tf.float32)
        dx_tf = model.time_derivative(x_tf) # (1, input_dim)
        dx = dx_tf.numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    # Stop if either (dtheta · rminus) < 0 or (dtheta · rplus) < 0
    return (np.dot(dtheta, rminus) < 0.0) or (np.dot(dtheta, rplus) < 0.0)

def build_tree(theta, r, logu, v, j, epsilon, joint0, call_lf, max_tree_depth=args.max_tree_length):
    """
    Recursively build the binary tree for NUTS:
    - If j == 0, take a single step.
    - Otherwise, build left and right subtrees.

    Args:
    theta : np.ndarray
        Position vector in the phase space of shape (D,).
    r : np.ndarray
        Momentum vector in the phase space of shape (D,).
    logu : float
        Logarithm of the slice variable u, where u ~ Uniform(0, exp(-H(theta, r))).
    v : int
        Direction of tree expansion: +1 for forward and -1 for backward integration.
    j : int
        Current depth of the tree. j = 0 represents the base case.
    epsilon : float
        Step size for integration in Hamiltonian dynamics.
    joint0 : float
        Hamiltonian value at the initial state of the current subtree.
    call_lf : bool
        Flag indicating whether to fallback to numerical leapfrog integration
    max_tree_depth : int, optional
        Maximum allowed depth of the tree (default is 30).

    Returns:
    -------
    thetaminus : np.ndarray
        Leftmost position vector of the subtree.
    rminus : np.ndarray
        Leftmost momentum vector of the subtree.
    thetaplus : np.ndarray
        Rightmost position vector of the subtree.
    rplus : np.ndarray
        Rightmost momentum vector of the subtree.
    thetaprime : np.ndarray
        Candidate position vector proposed during tree expansion.
    rprime : np.ndarray
        Candidate momentum vector proposed during tree expansion.
    nprime : int
        Number of valid samples generated within the subtree.
    sprime : int
        Subtree validity flag (1 if the subtree is valid, 0 otherwise).
    alphaprime : float
        Running sum

    """
    # Convert to np.float32 arrays for consistency
    theta = np.array(theta, dtype=np.float32)
    r     = np.array(r, dtype=np.float32)
    logu_val   = float(logu)
    joint0_val = float(joint0)

    # Safety limit on recursion
    if j > max_tree_depth:
        print("max tree depth reached")
        return (theta, r, theta, r, theta, r, 0, 0, 0., 0, 0., call_lf)

    # ---------------
    # BASE CASE
    # ---------------
    if j == 0:
        print("in base case")
        t_span1 = [0, epsilon]

        # Combine positions and momentum into y1
        y1 = np.concatenate((theta, r), axis=0)  # shape (input_dim,)

        # If v = -1, flip momentum in y1 before integration
        if v == -1:
            y1[D:] = -y1[D:]

        # Integrate using the HNN first (or partial HNN if you want)
        hnn_ivp1 = integrate_model(hnn_model, t_span1, y1, n=1)
        # hnn_ivp1.shape == (2, input_dim), where row 0 is initial, row 1 is final

        # Extract the final (theta, r)
        thetaprime = hnn_ivp1[1, 0:D]
        rprime = hnn_ivp1[1, D:args.input_dim]

        if v == -1:
            rprime = -rprime

        # Evaluate the Hamiltonian at the final state
        new_state = hnn_ivp1[1, :].reshape(1, args.input_dim)
        joint = functions(new_state)
        joint_val = float(joint.numpy()[0])

        monitor_val = (logu_val + joint_val)

        # Decide whether to fallback to numerical leapfrog if HNN error is too high
        # call_lf_bool = call_lf or (monitor_val > hnn_threshold)
        call_lf_bool = True
        sprime = int(monitor_val <= hnn_threshold)

        if call_lf_bool:
            print("switched to numerical leapfrog==============================")
            lf_ivp = leapfrog(dynamics_fn, t_span1, y1, 1, args.input_dim)

            # The final row of lf_ivp is the new state
            thetaprime = lf_ivp[1, 0:D]
            rprime = lf_ivp[1, D:args.input_dim]
            # Flip momentum back if v == -1
            if v == -1:
                rprime = -rprime

            new_state = lf_ivp[1, :].reshape(1, args.input_dim)
            joint = functions(new_state)
            joint_val = float(joint.numpy()[0])
            monitor_val = (logu_val + joint_val)
            sprime = int(monitor_val <= lf_threshold)

        # Acceptance info
        nprime = int(logu_val <= np.exp(-joint_val))
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        alphaprime = min(1., np.exp(joint0_val - joint_val))
        nalphaprime = 1
        call_lf = call_lf_bool

    # ---------------
    # RECURSIVE CASE
    # ---------------
    else:
        print("in ELSE case")
        # Build left subtree
        (thetaminus, rminus, thetaplus, rplus,
         thetaprime, rprime, nprime, sprime,
         alphaprime, nalphaprime, monitor_val,
         call_lf) = build_tree(theta, r, logu, v, j - 1, epsilon, joint0, call_lf)

        # If still continuing, build the right subtree
        if sprime == 1:
            if v == -1:
                (thetaminus, rminus, _, _, thetaprime2, rprime2,
                 nprime2, sprime2, alphaprime2, nalphaprime2,
                 monitor_val2, call_lf) = build_tree(
                    thetaminus, rminus, logu, v, j - 1, epsilon, joint0, call_lf
                )
            else:
                (_, _, thetaplus, rplus, thetaprime2, rprime2,
                 nprime2, sprime2, alphaprime2, nalphaprime2,
                 monitor_val2, call_lf) = build_tree(
                    thetaplus, rplus, logu, v, j - 1, epsilon, joint0, call_lf
                )

            # Combine results
            if (sprime2 == 1) and (
                np.random.uniform() < (float(nprime2) / max(float(nprime + nprime2), 1.))
            ):
                thetaprime = thetaprime2[:]
                rprime     = rprime2[:]

            nprime += nprime2
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            alphaprime += alphaprime2
            nalphaprime += nalphaprime2
            monitor_val = monitor_val2

    return (
        thetaminus, rminus, thetaplus, rplus,
        thetaprime, rprime, nprime, sprime,
        alphaprime, nalphaprime, monitor_val, call_lf
    )


#################################################################
##################### Actual Sampling ###########################
def main():
    D = int(args.input_dim / 2)
    M = N
    Madapt = 0
    theta0 = np.ones(D, dtype=np.float32)

    samples = np.empty((M + Madapt, D), dtype=float)
    samples[0, :] = theta0

    y0 = np.zeros(args.input_dim)
    # Initialize momenta with Normal(0,1)
    for ii in range(0, D):
        y0[ii] = np.random.randn()
    for ii in range(D, args.input_dim):
        y0[ii] = np.random.randn()

    HNN_accept = np.ones(M)
    traj_len = np.zeros(M)
    alpha_req = np.zeros(M)
    H_store = np.zeros(M)
    monitor_err = np.zeros(M)
    call_lf = 0
    counter_lf = 0
    is_lf = np.zeros(M)

    # print(samples)

    momentum_samples = np.empty((M + Madapt, D), dtype=float)
    momentum_samples[0, :] = y0[D:args.input_dim]

    for m in range(1, M + Madapt):
        print(m)
        # --------------------------------------------------
        # 1) Refresh momentum, build tree, accept new sample
        # --------------------------------------------------
        # (same as before)
        for ii in range(D, args.input_dim):
            y0[ii] = np.random.randn()

        joint = functions(y0[None, :])
        joint0_val = float(joint.numpy()[0])
        u_val = np.random.uniform(0, np.exp(-joint0_val))
        logu_val = np.log(u_val)

        samples[m, :] = samples[m - 1, :]
        momentum_samples[m, :] = momentum_samples[m - 1, :]

        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = y0[D:args.input_dim]
        rplus = y0[D:args.input_dim]

        j = 0
        n = 1
        s = 1
        if call_lf:
            counter_lf += 1
        if counter_lf == N_lf:
            call_lf = 0
            counter_lf = 0

        while s == 1:
            v = int(2 * (np.random.uniform() < 0.5) - 1)
            if v == -1:
                (thetaminus, rminus, _, _, thetaprime, rprime,
                nprime, sprime, alpha, nalpha, monitor, call_lf) = build_tree(
                    thetaminus, rminus, logu_val, v, j, epsilon, joint0_val, call_lf
                )
            else:
                (_, _, thetaplus, rplus, thetaprime, rprime,
                nprime, sprime, alpha, nalpha, monitor, call_lf) = build_tree(
                    thetaplus, rplus, logu_val, v, j, epsilon, joint0_val, call_lf
                )

            if (sprime == 1) and (np.random.uniform() < float(nprime) / float(n)):
                samples[m, :] = thetaprime[:]
                momentum_samples[m, :] = rprime[:]
                r_sto = rprime

            n += nprime
            s = sprime and (not stop_criterion(thetaminus, thetaplus, rminus, rplus))
            j += 1
            monitor_err[m] = monitor

        is_lf[m] = int(call_lf)
        traj_len[m] = j
        alpha_req[m] = alpha

        # Store final Hamiltonian
        y0[0:D] = samples[m, :]
        coords_sto = np.concatenate((samples[m, :], r_sto), axis=0).reshape(1, args.input_dim)
        H_val = functions(coords_sto)
        H_store[m] = float(H_val.numpy()[0])

    # ------------------- Plotting Distribution ----------------------------
    samples_lhnn = samples  # shape (M, D)

    # ------------------- Plot #1: L-HNN+NUTS histogram -------------------
    plt.figure(figsize=(7, 5))
    plt.hist(samples_lhnn[:, 0], bins=50, color='skyblue', edgecolor='k')
    plt.xlabel('theta[0]')
    plt.ylabel('Counts')
    plt.title('Histogram of theta[0] (L-HNN+NUTS)')
    plt.grid(True, alpha=0.3)
    # plt.savefig("hist_LHNN_NUTS.png", dpi=300)
    plt.savefig("hist_PureLeapfrog.png", dpi=300)
    plt.close()

class TestNUTSSampler(unittest.TestCase):
    def test_stop_criterion(self):
        thetaminus = np.array([0.0, 0.0])
        thetaplus  = np.array([1.0, 1.0])
        rminus     = np.array([ 0.5,  0.5])
        rplus      = np.array([-0.5, -0.5])
        result = stop_criterion(thetaminus, thetaplus, rminus, rplus)
        # (dtheta = [1,1]) dot rminus = 1, dot rplus = -1
        # This should return True if either < 0 => second dot < 0 => True
        self.assertTrue(result)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[''], exit=False)
    else:
        main()
