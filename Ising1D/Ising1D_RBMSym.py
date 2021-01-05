import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
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

pyfile = 'Ising1D_RBMSym'


def get_idx(l, i):
    return 3 * l + i


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
        "-L",
        "--length",
        type=int,
        help="Length of 1D spin chain"
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
        default=250,
        help="number of iterations for optimizing RBM (default=250)"
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
        default=0.05,
        help="learning rate of SGD optimizer (default=0.05)"
    )
    parser.add_argument(
        "-a",
        "--alpha_sym",
        type=int,
        default=1,
        help="hidden unit density for symmetrized RBM (default=1)"
    )
    parser.add_argument(
        "-ed",
        "--exact_diag",
        action="store_true",
        help="Compare results with exact diagonalization"
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

    L = args.length  # length of chain
    N = L  # number of sites
    J = args.coupling  # strength of the coupling
    h = args.htoJ * J  # strength of the transverse field
    n_iter_opt = args.n_iter_opt  # number of iterations for optimizing RBM
    n_iter_eval = args.n_iter_eval  # number of iterations for evaluating observables
    alpha_sym = args.alpha_sym  # hidden unit density for symmetrized RBM
    sgd_lr = args.sgd_lr  # learning rate of SGD optimizer
    logfile = open(args.log_filename, "a") \
        if args.log_filename != "" else None  # log file handler

    str_params = '_'.join(['N={}'.format(L),
                           'J={:.1f}'.format(J),
                           'h={:.1f}'.format(h),
                           'a={}'.format(alpha_sym),
                           'lr={:.3f}'.format(sgd_lr)])

    output("-----------------------------------------\n"
           "New execution. Local time: {0}.\n"
           ">>>> Params:\n"
           "     L={1}\n"
           "     J={2:.1f}\n"
           "     h={3:.1f}\n"
           "     n_iter_opt={4}\n"
           "     n_iter_eval={5}\n"
           "     alpha_sym={6:.1f}\n"
           "     sgd_lr={7:.4f}\n".format(
               time.asctime(time.localtime(time.time())),
               L, J, h, n_iter_opt, n_iter_eval, alpha_sym, sgd_lr),
           logfile)

    ###################################
    # Define Hilbert space, Hamiltonian
    ###################################

    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)  # 1D chain with pdc
    hi = nk.hilbert.Spin(s=0.5, graph=g)  # Hilbert space
    ha = nk.operator.Ising(hilbert=hi, J=J, h=h)  # Hamiltonian

    ################################
    # Exact Diagonalization: testbed
    ################################

    if args.exact_diag:
        res_ed = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
        gs_energy_ed = res_ed[0][0]
        gs_psi_ed = res_ed[1][:, 0]

    ######################################
    # Initialize RBM with lattice symmetry
    ######################################

    n_autom = len(g.automorphisms())
    ma = nk.machine.RbmSpinSymm(
        hilbert=hi, alpha=alpha_sym * n_autom // N)  # Machine
    ma.init_random_parameters(seed=1234, sigma=0.01)
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

    # debug
    # print("ma.n_par = ", ma.n_par)
    # print("ma.input_size = ", ma.input_size)
    # print("ma._as.shape = ", ma._as.shape)
    # print("ma._bs.shape = ", ma._bs.shape)
    # print("ma._ws.shape = ", ma._ws.shape)
    # print("ma._autom.shape[0] = ", ma._autom.shape[0])

    ################################################
    # Optimize RBM parameters to obtain ground state
    ################################################

    output(">>>> RBMSym optimization\n", logfile)

    start = time.time()
    gs.run(out="{}_opt_{}".format(pyfile, str_params),
           n_iter=n_iter_opt)
    end = time.time()

    output("     number of parameters = {}\n"
           "     number of iterations = {}\n"
           "     time consumption: {} seconds\n".format(
               ma.n_par, n_iter_opt, end - start),
           logfile)

    if args.exact_diag:

        # Import the data from log file
        data_opt = json.load(
            open("{}_opt_{}.log".format(pyfile, str_params)))
        # Extract the relevant information
        iters_opt = [it["Iteration"] for it in data_opt["Output"]]
        energy_opt = [it["Energy"]["Mean"] for it in data_opt["Output"]]

        # plot energy-iteration figure
        _, ax = plt.subplots()
        ax.plot(iters_opt, energy_opt, color='blue',
                label='Energy (Symmetric RBM)')
        ax.set_ylabel('Energy')
        ax.set_xlabel('Iteration')
        plt.axis([0, iters_opt[-1], gs_energy_ed - 0.06, gs_energy_ed + 0.12])
        plt.axhline(y=gs_energy_ed, xmin=0, xmax=iters_opt[-1],
                    linewidth=2, color='k', label='Energy (Exact)')
        ax.legend()
        plt.title('Symmetric RBM optimization')
        plt.savefig('{}_opt_{}.png'.format(pyfile, str_params))

    ##########################################
    # Evaluate observables using converged RBM
    ##########################################

    obs = {}

    # Add 1-local observables to the VMC object: sigma_i(0) (i=x,y,z)
    for i in range(3):
        obs[i] = nk.operator.LocalOperator(hilbert=hi,
                                           operators=pauli[i],
                                           acting_on=[0])

    # Add 2-local observables to the VMC object: sigma_i(0)*sigma_j(l)) (i,j=x,y,z, l=1,2,...,L/2)
    n_obser = 3
    for l, i, j in itertools.product(range(1, (L // 2) + 1), range(3), range(3)):
        obs[n_obser] = nk.operator.LocalOperator(hilbert=hi,
                                                 operators=pauliKron[i][j],
                                                 acting_on=[0, l])
        n_obser += 1

    output(">>>> Observable evaluation\n", logfile)

    start = time.time()
    gs.run(out="{}_eval_{}".format(pyfile, str_params),
           n_iter=n_iter_eval, obs=obs)
    end = time.time()

    output("     number of observables = {}\n"
           "     number of iterations = {}\n"
           "     time consumption: {} seconds\n".format(
               n_obser, n_iter_eval, end - start),
           logfile)

    # import the data from log file
    data_eval = json.load(
        open("{}_eval_{}.log".format(pyfile, str_params)))

    # Extract the relevant information
    iters_eval = [it["Iteration"] for it in data_eval["Output"]]
    energy_eval = [it["Energy"]["Mean"] for it in data_eval["Output"]]
    obser_eval = [[it[str(o_idx)]["Mean"] for o_idx in range(n_obser)]
                  for it in data_eval["Output"]]

    output("     energy = {0:.5f}({1:.5f})\n".format(np.mean(energy_eval),
                                                     np.std(energy_eval) / np.sqrt(n_iter_eval)),
           logfile)

    obser_avr = np.mean(np.asarray(obser_eval, dtype=np.complex), axis=0)

    # obser1_avr[i] = mean of sigma_i(0)
    obser1_avr = obser_avr[0:3]
    # obser1_dot_avr[i][j] = mean of sigma_i(0)*sigma_j(0)
    obser1_dot_avr = np.asarray([[permu_sgn[i][j] * 1j * obser1_avr[permu_idx[i][j]]
                                  if i != j else 1 for j in range(3)] for i in range(3)],
                                dtype=np.complex)
    # obser2_avr[l][i][j] = mean of sigma_i(0)*sigma_j(l+1)
    obser2_avr = obser_avr[3:].reshape([L // 2, 3, 3])

    # local1_avr[(l,i)] = mean of sigma_i(l)
    local1_avr = np.tile(obser1_avr, L)
    # local2_avr[(l,i),(k,j)] = mean of sigma_i(l)*sigma_j(k)
    local2_avr = np.empty([3 * L, 3 * L], dtype=np.complex)
    for l in range(L):
        local2_avr[3 * l:3 * l + 3, 3 * l:3 * l + 3] = obser1_dot_avr
        for dist in range(1, L // 2 + 1):
            k = (l + dist) % L
            local2_avr[3 * l:3 * l + 3, 3 * k:3 * k + 3] = obser2_avr[dist - 1]
        for dist in range(1, (L - 1) // 2 + 1):
            k = (l + L - dist) % L
            local2_avr[3 * l:3 * l + 3, 3 * k:3 * k +
                       3] = np.transpose(obser2_avr[dist - 1])

    VCM = local2_avr - np.outer(local1_avr, local1_avr)
    emax = max(np.real(eig(VCM)[0]))  # TODO: Use more efficient algo

    output("     emax(VCM) = {0:.5f}\n".format(emax), logfile)

    ##########################################
    # Compare with ed results.
    # ED is calculated without exploiting the symmetry property
    ##########################################

    if args.exact_diag:

        local1_ed = np.empty([3 * L], dtype=np.complex)
        for l, i in itertools.product(range(L), range(3)):
            idx = get_idx(l, i)
            local_op = nk.operator.LocalOperator(hilbert=hi,
                                                 operators=pauli[i],
                                                 acting_on=[l])
            local1_ed[idx] = get_mean(gs_psi_ed, local_op.to_sparse())

        local2_ed = np.empty([3 * L, 3 * L], dtype=np.complex)
        for l, k, i, j in itertools.product(range(L), range(L), range(3), range(3)):
            lidx, kidx = get_idx(l, i), get_idx(k, j)
            if not (l == k):
                local_op = nk.operator.LocalOperator(hilbert=hi,
                                                     operators=pauliKron[i][j],
                                                     acting_on=[l, k])
            else:
                local_op = nk.operator.LocalOperator(hilbert=hi,
                                                     operators=(
                                                         pauli[i] @ pauli[j]),
                                                     acting_on=[l])
            local2_ed[lidx, kidx] = get_mean(gs_psi_ed, local_op.to_sparse())

        VCM_ed = local2_ed - np.outer(local1_ed, local1_ed)
        emax_ed = max(np.real(eig(VCM_ed)[0]))  # TODO: Use more efficient algo

        output(">>>> Compare with ED results:\n", logfile)
        output("     exact ground-state energy E0 = {0:.5f}\n".format(
            gs_energy_ed), logfile)
        output("     emax(VCM_ed) = {0:.5f}\n".format(emax_ed), logfile)

    if logfile is not None:
        logfile.close()
