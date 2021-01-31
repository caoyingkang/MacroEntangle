import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import time
import argparse
import itertools
from scipy.linalg import eig

pauli = [np.asarray([[0, 1], [1, 0]]),
         np.asarray([[0, -1j], [1j, 0]]),
         np.asarray([[1, 0], [0, -1]])]
pauliKron = [[np.kron(pauli[i], pauli[j])
              for j in range(3)] for i in range(3)]

# For i!=j, pauli[i]*pauli[j] = permu_sgn[i][j] * 1j * pauli[permu_idx[i][j]]
permu_idx = [[0, 2, 1],
             [2, 1, 0],
             [1, 0, 2]]
permu_sgn = [[0, 1, -1],
             [-1, 0, 1],
             [1, -1, 0]]

pyfile = 'Ising3D_RBMSym'


def get_mean(psi, sp_op):
    return np.conj(psi) @ sp_op @ psi


def output(s, f=None):
    print(s)
    if f is not None:
        f.write(s)


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-L1",
        "--side1",
        type=int,
        default=0,
        help="Length of the first side of 3D spin lattice, i.e., total number of sites=L1*L2*L3"
    )
    parser.add_argument(
        "-L2",
        "--side2",
        type=int,
        default=0,
        help="Length of the second side of 3D spin lattice, i.e., total number of sites=L1*L2*L3"
    )
    parser.add_argument(
        "-L3",
        "--side3",
        type=int,
        default=0,
        help="Length of the third side of 3D spin lattice, i.e., total number of sites=L1*L2*L3"
    )
    parser.add_argument(
        "-J",
        "--coupling",
        type=float,
        default=1.0,
        help="strength of the spin coupling (default=1.0)"
    )
    parser.add_argument(
        "-g",
        "--htoJ",
        type=float,
        help="ratio of strength of the transverse field to the spin coupling"
    )
    parser.add_argument(
        "-t",
        "--n_iter_opt",
        type=int,
        default=300,
        help="number of iterations for optimizing RBM (default=300)"
    )
    parser.add_argument(
        "-n",
        "--n_iter_eval",
        type=int,
        default=50,
        help="number of iterations for evaluating observables (default=50)"
    )
    parser.add_argument(
        "-lr",
        "--sgd_lr",
        type=float,
        default=0.01,
        help="learning rate of SGD optimizer (default=0.01)"
    )
    parser.add_argument(
        "-a",
        "--alpha_sym",
        type=int,
        default=1,
        help="hidden unit density for symmetrized RBM (default=1)"
    )
    parser.add_argument(
        "-dc",
        "--direct_calc",
        action="store_true",
        help="Skip the training of RBM, and directly calculate the desired value"
    )
    parser.add_argument(
        "-log",
        "--log_filename",
        type=str,
        default="",
        help="path to the log file (append logs to it, default: no log file)"
    )
    args = parser.parse_args()

    ####################
    # Tunable parameters
    ####################

    L1, L2, L3 = args.side1, args.side2, args.side3  # side length of lattice
    N = L1 * L2 * L3  # number of sites
    J = args.coupling  # strength of the coupling
    h = args.htoJ * J  # strength of the transverse field
    n_iter_opt = args.n_iter_opt  # number of iterations for optimizing RBM
    n_iter_eval = args.n_iter_eval  # number of iterations for evaluating observables
    alpha_sym = args.alpha_sym  # hidden unit density for symmetrized RBM
    sgd_lr = args.sgd_lr  # learning rate of SGD optimizer
    logfile = open(args.log_filename, "a") \
        if args.log_filename != "" else None  # log file handler

    str_params = '_'.join(['N={}x{}x{}'.format(L1, L2, L3),
                           'J={:.1f}'.format(J),
                           'h={:.1f}'.format(h),
                           'a={}'.format(alpha_sym),
                           'lr={:.3f}'.format(sgd_lr)])

    output("-----------------------------------------\n"
           "New execution. Local time: {0}.\n"
           ">>>> Params:\n"
           "     N={1}x{2}x{3}\n"
           "     J={4:.1f}\n"
           "     h={5:.1f}\n"
           "     n_iter_opt={6}\n"
           "     n_iter_eval={7}\n"
           "     alpha_sym={8:.1f}\n"
           "     sgd_lr={9:.4f}\n".format(
               time.asctime(time.localtime(time.time())),
               L1, L2, L3, J, h, n_iter_opt, n_iter_eval, alpha_sym, sgd_lr),
           logfile)

    def get_site(x, y, z):
        return x + y * L1 + z * L1 * L2

    ###################################
    # Define Hilbert space, Hamiltonian
    ###################################

    symbrk_scale = 0.02

    if J < 0:  # ferromagnetic

        g = nk.graph.Grid(length=[L1, L2, L3], pbc=True)  # 3D lattice with pdc
        # note that for 0<=x<L1, 0<=y<L2, 0<=z<L3, the site
        # (x,y,z) is numbered x+y*L1+z*L1*L2
        hi = nk.hilbert.Spin(s=0.5, graph=g)  # Hilbert space
        ha = nk.operator.LocalOperator(hi)  # Hamiltonian
        for x, y, z in itertools.product(range(L1), range(L2), range(L3)):
            s = get_site(x, y, z)
            sr = get_site((x+1) % L1, y, z)
            sf = get_site(x, (y+1) % L2, z)
            su = get_site(x, y, (z+1) % L3)
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=J*pauliKron[2][2],
                                            acting_on=[s, sr])
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=J*pauliKron[2][2],
                                            acting_on=[s, sf])
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=J*pauliKron[2][2],
                                            acting_on=[s, su])
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=-h*pauli[0],
                                            acting_on=[s])
        # add a small perturbation in order for spontaneous symmetry breaking
        pert_op = symbrk_scale * J * pauli[2]
        for x, y, z in itertools.product(range(L1), range(L2), range(L3)):
            s = get_site(x, y, z)
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=pert_op,
                                            acting_on=[s])

    else:  # anti-ferromagnetic, add stagger perturbation

        assert L1 % 2 == 0 and L2 % 2 == 0 and L3 % 2 == 0
        g = nk.graph.Grid(length=[L1, L2, L3], pbc=True)  # 3D lattice with pdc
        # note that for 0<=x<L1, 0<=y<L2, 0<=z<L3, the site
        # (x,y,z) is numbered x+y*L1+z*L1*L2
        g._automorphisms = \
            [[get_site((x+dx) % L1, (y+dy) % L2, (z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))] + \
            [[get_site((L1-x+dx) % L1, (y+dy) % L2, (z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))] + \
            [[get_site((x+dx) % L1, (L2-y+dy) % L2, (z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))] + \
            [[get_site((L1-x+dx) % L1, (L2-y+dy) % L2, (z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))] + \
            [[get_site((x+dx) % L1, (y+dy) % L2, (L3-z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))] + \
            [[get_site((L1-x+dx) % L1, (y+dy) % L2, (L3-z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))] + \
            [[get_site((x+dx) % L1, (L2-y+dy) % L2, (L3-z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))] + \
            [[get_site((L1-x+dx) % L1, (L2-y+dy) % L2, (L3-z+dz) % L3)
              for z, y, x in itertools.product(range(L3), range(L2), range(L1))]
             for dz, dy, dx in itertools.product(range(0, L3, 2), range(0, L2, 2), range(0, L1, 2))]
        hi = nk.hilbert.Spin(s=0.5, graph=g)  # Hilbert space
        ha = nk.operator.LocalOperator(hi)  # Hamiltonian
        for x, y, z in itertools.product(range(L1), range(L2), range(L3)):
            s = get_site(x, y, z)
            sr = get_site((x+1) % L1, y, z)
            sf = get_site(x, (y+1) % L2, z)
            su = get_site(x, y, (z+1) % L3)
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=J*pauliKron[2][2],
                                            acting_on=[s, sr])
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=J*pauliKron[2][2],
                                            acting_on=[s, sf])
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=J*pauliKron[2][2],
                                            acting_on=[s, su])
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=-h*pauli[0],
                                            acting_on=[s])
        # add a small perturbation in order for spontaneous symmetry breaking
        pert_op = -symbrk_scale * J * pauli[2]
        for x, y, z in itertools.product(range(L1), range(L2), range(L3)):
            s = get_site(x, y, z)
            ha += nk.operator.LocalOperator(hilbert=hi,
                                            operators=(pert_op if (x+y+z) %
                                                       2 == 0 else -pert_op),
                                            acting_on=[s])

    ######################################
    # Initialize RBM with lattice symmetry
    ######################################

    n_autom = len(g.automorphisms())
    ma = nk.machine.RbmSpinSymm(
        hilbert=hi, alpha=alpha_sym * n_autom // N)  # Machine
    ma.init_random_parameters(seed=1234, sigma=0.01)

    print("ma._ws.shape =", ma._ws.shape)

    assert(ma._ws.shape == (N, alpha_sym))

    opt = nk.optimizer.Sgd(ma, learning_rate=sgd_lr)  # Optimizer
    sa = nk.sampler.MetropolisLocal(machine=ma)  # Metropolis Local Sampling
    sr = nk.optimizer.SR(ma, diag_shift=0.1)  # Stochastic Reconfiguration
    gs = nk.Vmc(  # VMC object
        hamiltonian=ha,
        sampler=sa,
        optimizer=opt,
        n_samples=1000,
        sr=sr)

    ################################################
    # Optimize RBM parameters to obtain ground state
    ################################################

    if args.direct_calc:
        assert os.path.isfile("data/{}_opt_{}.log".format(pyfile, str_params))
        output(">>>> RBMSym optimization skipped\n", logfile)
    else:
        output(">>>> RBMSym optimization\n", logfile)

        start = time.time()
        gs.run(out="data/{}_opt_{}".format(pyfile, str_params),
               n_iter=n_iter_opt)
        end = time.time()

        output("     number of parameters = {}\n"
               "     number of iterations = {}\n"
               "     time consumption: {} seconds\n".format(
                   ma.n_par, n_iter_opt, end - start),
               logfile)

    ##########################################
    # Evaluate observables using converged RBM
    ##########################################

    op_sz = nk.operator.LocalOperator(hilbert=hi,
                                      operators=pauli[2],
                                      acting_on=[0])
    obs = {"Localsz": op_sz}

    if args.direct_calc:
        assert os.path.isfile("data/{}_eval_{}.log".format(pyfile, str_params))
        output(">>>> Observable evaluation skipped\n", logfile)
    else:
        output(">>>> Observable evaluation\n", logfile)

        start = time.time()
        gs.run(out="data/{}_eval_{}".format(pyfile, str_params),
               n_iter=n_iter_eval, obs=obs)
        end = time.time()

        output("     number of observables = {}\n"
               "     number of iterations = {}\n"
               "     time consumption: {} seconds\n".format(
                   len(obs), n_iter_eval, end - start),
               logfile)

    # raw data collected, now extract the relevant information
    data_eval = json.load(
        open("data/{}_eval_{}.log".format(pyfile, str_params)))
    assert n_iter_eval == len(data_eval["Output"])

    # Extract the relevant information
    iters_eval = [it["Iteration"] for it in data_eval["Output"]]
    energy_eval = [it["Energy"]["Mean"] for it in data_eval["Output"]]
    sz_eval = [it["Localsz"]["Mean"] for it in data_eval["Output"]]

    output("     energy = {0:.5f}({1:.5f})\n".format(np.mean(energy_eval),
                                                     np.std(energy_eval) / np.sqrt(n_iter_eval)),
           logfile)
    output("     localsz = {0:.5f}({1:.5f})\n".format(np.mean(sz_eval),
                                                      np.std(sz_eval) / np.sqrt(n_iter_eval)),
           logfile)

    if logfile is not None:
        logfile.close()
