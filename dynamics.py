from __future__ import print_function
import math
import numpy as np
import torch
from torch.distributions import MultivariateNormal


class DynamicsConfig:
    num_agent = 128
    num_policy = 128
    num_step = 50
    norm_policy_noise = 1e-3


class DynamicsModel(DynamicsConfig):

    def __init__(self):

        super(DynamicsModel, self).__init__()
        # dimension of state, control and disturbance
        self.num_state = 2
        self.num_observation = 1
        self.num_control = 1

        # parameters of discrete dynamics
        self.A = torch.tensor([[1.1, -0.3],
                               [1.0,  0.0]], dtype=torch.float64)
        self.B = torch.tensor([[1], [0]], dtype=torch.float64)
        self.C = torch.tensor([[0.5, -0.4]], dtype=torch.float64)

        # perform singular value decomposition on the output matrix C
        self.U, self.Y_d, self.VT = torch.linalg.svd(self.C)
        self.Y = torch.cat((torch.diag_embed(self.Y_d), torch.zeros(self.num_observation, self.num_state - self.num_observation)), dim=1)
        self.V = self.VT.t()
        print(f'error of svd: {torch.norm(self.C - torch.mm(torch.mm(self.U, self.Y), self.V.t())):.2f}')
        print(f'error of I: {torch.norm(torch.eye(self.num_state) - torch.mm(self.V, self.V.t())):.2f}, '
              f'{torch.norm(torch.eye(self.num_state) - torch.mm(self.V.t(), self.V)):.2f}, '
              f'{torch.norm(torch.eye(self.num_observation) - torch.mm(self.U, self.U.t())):.2f}, '
              f'{torch.norm(torch.eye(self.num_observation) - torch.mm(self.U.t(), self.U)):.2f}')

        # partition the unitary matrix V and the singular value matrix S
        self.V1 = self.V[:, 0:self.num_observation]
        self.V2 = self.V[:, self.num_observation:self.num_state]
        self.S1 = self.Y[:, 0:self.num_observation]
        self.C_pseudo_inv = torch.mm(torch.mm(self.V1, torch.inverse(self.S1)), self.U.t())
        print(f'error of partition: '
              f'{torch.norm(self.C - torch.mm(torch.mm(self.U, self.S1), self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.C_pseudo_inv, self.C) - torch.mm(self.V1, self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.V2.t(), self.V1)):.2f}, '
              f'{torch.norm(torch.eye(self.num_state) - torch.mm(self.V1, self.V1.t()) - torch.mm(self.V2, self.V2.t())):.2f}')

        # weighting matrices of utility function
        self.Q_y = torch.tensor([[1.]], dtype=torch.float64)
        self.Q = torch.mm(torch.mm(self.C.t(), self.Q_y), self.C)
        self.R = torch.tensor([[0.2]], dtype=torch.float64)
        self.X = torch.eye(self.num_state, dtype=torch.float64)
        self.I_num_state = torch.eye(self.num_state, dtype=torch.float64)
        self.Sigma = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)

        # the optimal parameter of networks
        self.opt_policy = torch.tensor([[0.374609149867465]], dtype=torch.float64)
        self.init_policy = torch.tensor([[0.0]], dtype=torch.float64)
        self.K0 = torch.mm(self.init_policy, self.C)

    def random_initialize(self):

        distribution = MultivariateNormal(torch.zeros(self.num_state), torch.eye(self.num_state))
        state = distribution.sample(sample_shape=torch.Size([self.num_agent])).double()  # torch.Size([1, 2])

        return state

    def policy_noise(self):

        theta = torch.rand(1, )
        U = torch.zeros_like(self.init_policy, dtype=torch.float64)

        if theta <= 0.5:
            U[0, 0] = -1.0
        else:
            U[0, 0] = 1.0

        return U

    def rollout_n_step(self, initial_state, F_hat):

        accumulated_cost = torch.zeros([self.num_agent, 1], dtype=torch.float64)
        accumulated_corr = torch.zeros([self.num_agent, self.num_observation, self.num_observation], dtype=torch.float64)

        state = initial_state  # torch.Size([batch_size, 4])
        for i in range(self.num_step):
            observation = torch.mm(state, self.C.t())    # torch.Size([batch_size, 1])
            control = -torch.mm(observation, F_hat.t())  # torch.Size([batch_size, 1])
            state_next = torch.mm(state, self.A.t()) + torch.mm(control, self.B.t())  # torch.Size([batch_size, 2])

            cost = self.cost_func(state, control)
            correlation = torch.bmm(observation[:, :, np.newaxis], observation[:, np.newaxis, :])

            accumulated_cost += cost         # torch.Size([batch_size, 1])
            accumulated_corr += correlation  # torch.Size([batch_size, 1, 1])
            state = state_next

        return accumulated_cost, accumulated_corr

    def cost_func(self, state, control):

        cost = torch.bmm(torch.mm(state, self.Q)[:, np.newaxis, :], state[:, :, np.newaxis]).squeeze(-1) \
               + torch.bmm(torch.mm(control, self.R)[:, np.newaxis, :], control[:, :, np.newaxis]).squeeze(-1)

        return cost

    @staticmethod
    def dlyap(A, Q):
        """
        A X A ^ T - X + Q = 0
        X = dlyap(A, Q)
        """
        assert max(torch.abs(torch.linalg.eigvals(A))).item() < 1.0, 'unstable closed-loop dynamic system'
        Ar = A.t()
        a1 = Ar[0, 0]
        a2 = Ar[0, 1]
        a3 = Ar[1, 0]
        a4 = Ar[1, 1]

        M = torch.tensor([[a1**2 - 1, 2 * a1 * a3, a3**2],
                          [a1 * a2, a1 * a4 + a2 * a3 - 1, a3 * a4],
                          [a2**2, 2 * a2 * a4, a4**2 - 1]], dtype=torch.float64)
        N = torch.tensor([[Q[0, 0].item()], [Q[0, 1].item()], [Q[1, 1].item()]], dtype=torch.float64)
        p = torch.mm(torch.inverse(M), -N)

        X = torch.tensor([[0, 0], [0, 0]], dtype=torch.float64)
        X[0, 0] = p[0, 0]
        X[0, 1] = p[1, 0]
        X[1, 0] = p[1, 0]
        X[1, 1] = p[2, 0]
        return X

    def lyap(self, K):
        """
        (A - B K)^T P_K (A - B K) - P_K + Q + K^T R K = 0, or
        (A - B F C)^T P_K (A - B F C) - P_K + Q + C^T F^T R F C = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution of Lyapunov Equation
        """
        Ar = self.A - torch.mm(self.B, K)
        P_K = self.dlyap(Ar.t(), self.Q + torch.mm(torch.mm(K.t(), self.R), K))
        return P_K

    def obj_func(self, K):
        """
        J(K) = Tr(P_K X_0) = Tr(P_K Sigma)
        :param K: F * C, where F is the output feedback gain
        :return: the objective function of F * C
        """
        P_K = self.lyap(K)
        J_K = torch.mm(P_K, self.Sigma).trace()
        return J_K

    def state_correlation(self, K):
        """
        (A - B K) Sigma_K (A - B K)^T - Sigma_K + I = 0, or
        (A - B F C) Sigma_K (A - B F C)^T - Sigma_K + I = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution
        """
        Ar = self.A - torch.mm(self.B, K)
        Sigma_K = self.dlyap(Ar, torch.eye(self.num_state, dtype=torch.float64))
        return Sigma_K


class Doyle(DynamicsConfig):

    def __init__(self):

        super(Doyle, self).__init__()
        # dimension of state, control and disturbance
        self.num_state = 2
        self.num_observation = 1
        self.num_control = 1

        # parameters of discrete dynamics
        self.A = torch.tensor([[1.1, 0.1],
                               [0.0, 1.1]], dtype=torch.float64)
        self.B = torch.tensor([[0.0], [0.1]], dtype=torch.float64)
        self.C = torch.tensor([[1.0, 1.0]], dtype=torch.float64)

        # perform singular value decomposition on the output matrix C
        self.U, self.Y_d, self.VT = torch.linalg.svd(self.C)
        self.Y = torch.cat((torch.diag_embed(self.Y_d), torch.zeros(self.num_observation, self.num_state - self.num_observation)), dim=1)
        self.V = self.VT.t()
        print(f'error of svd: {torch.norm(self.C - torch.mm(torch.mm(self.U, self.Y), self.V.t())):.2f}')
        print(f'error of I: {torch.norm(torch.eye(self.num_state) - torch.mm(self.V, self.V.t())):.2f}, '
              f'{torch.norm(torch.eye(self.num_state) - torch.mm(self.V.t(), self.V)):.2f}, '
              f'{torch.norm(torch.eye(self.num_observation) - torch.mm(self.U, self.U.t())):.2f}, '
              f'{torch.norm(torch.eye(self.num_observation) - torch.mm(self.U.t(), self.U)):.2f}')

        # partition the unitary matrix V and the singular value matrix S
        self.V1 = self.V[:, 0:self.num_observation]
        self.V2 = self.V[:, self.num_observation:self.num_state]
        self.S1 = self.Y[:, 0:self.num_observation]
        self.C_pseudo_inv = torch.mm(torch.mm(self.V1, torch.inverse(self.S1)), self.U.t())
        print(f'error of partition: '
              f'{torch.norm(self.C - torch.mm(torch.mm(self.U, self.S1), self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.C_pseudo_inv, self.C) - torch.mm(self.V1, self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.V2.t(), self.V1)):.2f}, '
              f'{torch.norm(torch.eye(self.num_state) - torch.mm(self.V1, self.V1.t()) - torch.mm(self.V2, self.V2.t())):.2f}')

        # weighting matrices of utility function
        self.Q = torch.tensor([[5., 5.], [5., 5.]], dtype=torch.float64)
        self.R = torch.tensor([[1.0]], dtype=torch.float64)
        self.X = torch.eye(self.num_state, dtype=torch.float64)
        self.I_num_state = torch.eye(self.num_state, dtype=torch.float64)
        self.Sigma = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)

        # the optimal parameter of networks
        self.opt_policy = torch.tensor([[4.670022407193215]], dtype=torch.float64)
        self.init_policy = torch.tensor([[5.0]], dtype=torch.float64)
        self.K0 = torch.mm(self.init_policy, self.C)

    def random_initialize(self):

        distribution = MultivariateNormal(torch.zeros(self.num_state), torch.eye(self.num_state))
        state = distribution.sample(sample_shape=torch.Size([self.num_agent])).double()  # torch.Size([1, 2])

        return state

    def policy_noise(self):

        theta = torch.rand(1, )
        U = torch.zeros_like(self.init_policy, dtype=torch.float64)

        if theta <= 0.5:
            U[0, 0] = -1.0
        else:
            U[0, 0] = 1.0

        return U

    def rollout_n_step(self, initial_state, F_hat):

        accumulated_cost = torch.zeros([self.num_agent, 1], dtype=torch.float64)
        accumulated_corr = torch.zeros([self.num_agent, self.num_observation, self.num_observation], dtype=torch.float64)

        state = initial_state  # torch.Size([batch_size, 4])
        for i in range(self.num_step):
            observation = torch.mm(state, self.C.t())    # torch.Size([batch_size, 1])
            control = -torch.mm(observation, F_hat.t())  # torch.Size([batch_size, 1])
            state_next = torch.mm(state, self.A.t()) + torch.mm(control, self.B.t())  # torch.Size([batch_size, 2])

            cost = self.cost_func(state, control)
            correlation = torch.bmm(observation[:, :, np.newaxis], observation[:, np.newaxis, :])

            accumulated_cost += cost         # torch.Size([batch_size, 1])
            accumulated_corr += correlation  # torch.Size([batch_size, 1, 1])
            state = state_next

        return accumulated_cost, accumulated_corr

    def cost_func(self, state, control):

        cost = torch.bmm(torch.mm(state, self.Q)[:, np.newaxis, :], state[:, :, np.newaxis]).squeeze(-1) \
               + torch.bmm(torch.mm(control, self.R)[:, np.newaxis, :], control[:, :, np.newaxis]).squeeze(-1)

        return cost

    @staticmethod
    def dlyap(A, Q):
        """
        A X A ^ T - X + Q = 0
        X = dlyap(A, Q)
        """
        assert max(torch.abs(torch.linalg.eigvals(A))).item() < 1.0, 'unstable closed-loop dynamic system'
        Ar = A.t()
        a1 = Ar[0, 0]
        a2 = Ar[0, 1]
        a3 = Ar[1, 0]
        a4 = Ar[1, 1]

        M = torch.tensor([[a1**2 - 1, 2 * a1 * a3, a3**2],
                          [a1 * a2, a1 * a4 + a2 * a3 - 1, a3 * a4],
                          [a2**2, 2 * a2 * a4, a4**2 - 1]], dtype=torch.float64)
        N = torch.tensor([[Q[0, 0].item()], [Q[0, 1].item()], [Q[1, 1].item()]], dtype=torch.float64)
        p = torch.mm(torch.inverse(M), -N)

        X = torch.tensor([[0, 0], [0, 0]], dtype=torch.float64)
        X[0, 0] = p[0, 0]
        X[0, 1] = p[1, 0]
        X[1, 0] = p[1, 0]
        X[1, 1] = p[2, 0]
        return X

    def lyap(self, K):
        """
        (A - B K)^T P_K (A - B K) - P_K + Q + K^T R K = 0, or
        (A - B F C)^T P_K (A - B F C) - P_K + Q + C^T F^T R F C = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution of Lyapunov Equation
        """
        Ar = self.A - torch.mm(self.B, K)
        P_K = self.dlyap(Ar.t(), self.Q + torch.mm(torch.mm(K.t(), self.R), K))
        return P_K

    def obj_func(self, K):
        """
        J(K) = Tr(P_K X_0) = Tr(P_K Sigma)
        :param K: F * C, where F is the output feedback gain
        :return: the objective function of F * C
        """
        P_K = self.lyap(K)
        J_K = torch.mm(P_K, self.Sigma).trace()
        return J_K

    def state_correlation(self, K):
        """
        (A - B K) Sigma_K (A - B K)^T - Sigma_K + I = 0, or
        (A - B F C) Sigma_K (A - B F C)^T - Sigma_K + I = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution
        """
        Ar = self.A - torch.mm(self.B, K)
        Sigma_K = self.dlyap(Ar, torch.eye(self.num_state, dtype=torch.float64))
        return Sigma_K


class Lqg(DynamicsConfig):

    def __init__(self):

        super(Lqg, self).__init__()
        # dimension of state, control and disturbance
        self.num_state = 2
        self.num_observation = 1
        self.num_control = 1

        # parameters of discrete dynamics
        self.A = torch.tensor([[1.0, 0.05],
                               [0.0, 1.0]], dtype=torch.float64)
        self.B = torch.tensor([[0], [0.05]], dtype=torch.float64)
        self.C = torch.tensor([[1, 0]], dtype=torch.float64)

        # perform singular value decomposition on the output matrix C
        self.U, self.Y_d, self.VT = torch.linalg.svd(self.C)
        self.Y = torch.cat((torch.diag_embed(self.Y_d), torch.zeros(self.num_observation, self.num_state - self.num_observation)), dim=1)
        self.V = self.VT.t()
        print(f'error of svd: {torch.norm(self.C - torch.mm(torch.mm(self.U, self.Y), self.V.t())):.2f}')
        print(f'error of I: {torch.norm(torch.eye(self.num_state) - torch.mm(self.V, self.V.t())):.2f}, '
              f'{torch.norm(torch.eye(self.num_state) - torch.mm(self.V.t(), self.V)):.2f}, '
              f'{torch.norm(torch.eye(self.num_observation) - torch.mm(self.U, self.U.t())):.2f}, '
              f'{torch.norm(torch.eye(self.num_observation) - torch.mm(self.U.t(), self.U)):.2f}')

        # partition the unitary matrix V and the singular value matrix S
        self.V1 = self.V[:, 0:self.num_observation]
        self.V2 = self.V[:, self.num_observation:self.num_state]
        self.S1 = self.Y[:, 0:self.num_observation]
        self.C_pseudo_inv = torch.mm(torch.mm(self.V1, torch.inverse(self.S1)), self.U.t())
        print(f'error of partition: '
              f'{torch.norm(self.C - torch.mm(torch.mm(self.U, self.S1), self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.C_pseudo_inv, self.C) - torch.mm(self.V1, self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.V2.t(), self.V1)):.2f}, '
              f'{torch.norm(torch.eye(self.num_state) - torch.mm(self.V1, self.V1.t()) - torch.mm(self.V2, self.V2.t())):.2f}')

        # weighting matrices of utility function
        self.Q = torch.tensor([[5., 5.], [5., 5.]], dtype=torch.float64)
        self.R = torch.tensor([[1.0]], dtype=torch.float64)
        self.X = torch.tensor([[0.2, 0.], [0., 0.8]], dtype=torch.float64)
        self.I_num_state = torch.eye(self.num_state, dtype=torch.float64)
        self.Sigma = torch.tensor([[0.2, 0.], [0., 0.8]], dtype=torch.float64)

        # the optimal parameter of networks
        self.opt_policy = torch.tensor([[-2.003278692398220]], dtype=torch.float64)
        self.init_policy = torch.tensor([[-1.0]], dtype=torch.float64)
        self.K0 = torch.mm(self.init_policy, self.C)

    def random_initialize(self):

        distribution = MultivariateNormal(torch.zeros(self.num_state), torch.eye(self.num_state))
        state = distribution.sample(sample_shape=torch.Size([self.num_agent])).double()  # torch.Size([1, 2])

        return state

    def policy_noise(self):

        theta = torch.rand(1, )
        U = torch.zeros_like(self.init_policy, dtype=torch.float64)

        if theta <= 0.5:
            U[0, 0] = -1.0
        else:
            U[0, 0] = 1.0

        return U

    def rollout_n_step(self, initial_state, F_hat):

        accumulated_cost = torch.zeros([self.num_agent, 1], dtype=torch.float64)
        accumulated_corr = torch.zeros([self.num_agent, self.num_observation, self.num_observation], dtype=torch.float64)

        state = initial_state  # torch.Size([batch_size, 4])
        for i in range(self.num_step):
            observation = torch.mm(state, self.C.t())    # torch.Size([batch_size, 1])
            control = -torch.mm(observation, F_hat.t())  # torch.Size([batch_size, 1])
            state_next = torch.mm(state, self.A.t()) + torch.mm(control, self.B.t())  # torch.Size([batch_size, 2])

            cost = self.cost_func(state, control)
            correlation = torch.bmm(observation[:, :, np.newaxis], observation[:, np.newaxis, :])

            accumulated_cost += cost         # torch.Size([batch_size, 1])
            accumulated_corr += correlation  # torch.Size([batch_size, 1, 1])
            state = state_next

        return accumulated_cost, accumulated_corr

    def cost_func(self, state, control):

        cost = torch.bmm(torch.mm(state, self.Q)[:, np.newaxis, :], state[:, :, np.newaxis]).squeeze(-1) \
               + torch.bmm(torch.mm(control, self.R)[:, np.newaxis, :], control[:, :, np.newaxis]).squeeze(-1)

        return cost

    @staticmethod
    def dlyap(A, Q):
        """
        A X A ^ T - X + Q = 0
        X = dlyap(A, Q)
        """
        assert max(torch.abs(torch.linalg.eigvals(A))).item() < 1.0, 'unstable closed-loop dynamic system'
        Ar = A.t()
        a1 = Ar[0, 0]
        a2 = Ar[0, 1]
        a3 = Ar[1, 0]
        a4 = Ar[1, 1]

        M = torch.tensor([[a1**2 - 1, 2 * a1 * a3, a3**2],
                          [a1 * a2, a1 * a4 + a2 * a3 - 1, a3 * a4],
                          [a2**2, 2 * a2 * a4, a4**2 - 1]], dtype=torch.float64)
        N = torch.tensor([[Q[0, 0].item()], [Q[0, 1].item()], [Q[1, 1].item()]], dtype=torch.float64)
        p = torch.mm(torch.inverse(M), -N)

        X = torch.tensor([[0, 0], [0, 0]], dtype=torch.float64)
        X[0, 0] = p[0, 0]
        X[0, 1] = p[1, 0]
        X[1, 0] = p[1, 0]
        X[1, 1] = p[2, 0]
        return X

    def lyap(self, K):
        """
        (A - B K)^T P_K (A - B K) - P_K + Q + K^T R K = 0, or
        (A - B F C)^T P_K (A - B F C) - P_K + Q + C^T F^T R F C = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution of Lyapunov Equation
        """
        Ar = self.A - torch.mm(self.B, K)
        P_K = self.dlyap(Ar.t(), self.Q + torch.mm(torch.mm(K.t(), self.R), K))
        return P_K

    def obj_func(self, K):
        """
        J(K) = Tr(P_K X_0) = Tr(P_K Sigma)
        :param K: F * C, where F is the output feedback gain
        :return: the objective function of F * C
        """
        P_K = self.lyap(K)
        J_K = torch.mm(P_K, self.Sigma).trace()
        return J_K

    def state_correlation(self, K):
        """
        (A - B K) Sigma_K (A - B K)^T - Sigma_K + I = 0, or
        (A - B F C) Sigma_K (A - B F C)^T - Sigma_K + I = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution
        """
        Ar = self.A - torch.mm(self.B, K)
        Sigma_K = self.dlyap(Ar, torch.eye(self.num_state, dtype=torch.float64))
        return Sigma_K


class Circuit(DynamicsConfig):

    def __init__(self):

        super(Circuit, self).__init__()
        # dimension of state, control and disturbance
        self.num_state = 4
        self.num_observation = 2
        self.num_control = 2

        # parameters of discrete dynamics
        self.A = torch.tensor([[ 0.90031, -0.00015,  0.09048, -0.00452],
                               [-0.00015,  0.90031,  0.00452, -0.09048],
                               [-0.09048, -0.00452,  0.90483, -0.09033],
                               [ 0.00452,  0.09048, -0.09033,  0.90483]], dtype=torch.float64)
        self.B = torch.tensor([[ 0.00468, -0.00015],
                               [ 0.00015, -0.00468],
                               [ 0.09516, -0.00467],
                               [-0.00467,  0.09516]], dtype=torch.float64)
        self.C = torch.tensor([[1, 1, 0, 0],
                               [0, 1, 0, 0]], dtype=torch.float64)

        # perform singular value decomposition on the output matrix C
        self.U, self.Y_d, self.VT = torch.linalg.svd(self.C)
        self.Y = torch.cat((torch.diag_embed(self.Y_d), torch.zeros(self.num_observation, self.num_state - self.num_observation)), dim=1)
        self.V = self.VT.t()
        print(f'error of svd: {math.log10(torch.norm(self.C - torch.mm(torch.mm(self.U, self.Y), self.V.t()))):.2f}')
        print(f'error of I: {math.log10(torch.norm(torch.eye(self.num_state) - torch.mm(self.V, self.V.t()))):.2f}, '
              f'{math.log10(torch.norm(torch.eye(self.num_state) - torch.mm(self.V.t(), self.V))):.2f}, '
              f'{math.log10(torch.norm(torch.eye(self.num_observation) - torch.mm(self.U, self.U.t()))):.2f}, '
              f'{math.log10(torch.norm(torch.eye(self.num_observation) - torch.mm(self.U.t(), self.U))):.2f}')

        # partition the unitary matrix V and the singular value matrix S
        self.V1 = self.V[:, 0:self.num_observation]
        self.V2 = self.V[:, self.num_observation:self.num_state]
        self.S1 = self.Y[:, 0:self.num_observation]
        self.C_pseudo_inv = torch.mm(torch.mm(self.V1, torch.inverse(self.S1)), self.U.t())
        print(f'error of partition: '
              f'{torch.norm(self.C - torch.mm(torch.mm(self.U, self.S1), self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.C_pseudo_inv, self.C) - torch.mm(self.V1, self.V1.t())):.2f}, '
              f'{torch.norm(torch.mm(self.V2.t(), self.V1)):.2f}, '
              f'{torch.norm(torch.eye(self.num_state) - torch.mm(self.V1, self.V1.t()) - torch.mm(self.V2, self.V2.t())):.2f}')

        # weighting matrices of utility function
        self.Q = torch.eye(self.num_state, dtype=torch.float64)
        self.Q[0, 0] = 0.1
        self.Q[1, 1] = 0.2
        self.Q[2, 2] = 0.0
        self.Q[3, 3] = 0.0
        self.R = torch.eye(self.num_control, dtype=torch.float64)
        self.R[0, 0] = 1e-6
        self.R[1, 1] = 1e-4
        self.X = 10 * torch.eye(self.num_state, dtype=torch.float64)
        self.I_num_state = torch.eye(self.num_state, dtype=torch.float64)
        self.Sigma = torch.eye(self.num_state, dtype=torch.float64)

        # the optimal parameter of networks
        self.opt_policy = torch.tensor([[2.973788024461090, -7.290737922994562],
                                        [2.106731464803984, -12.538401840035132]], dtype=torch.float64)
        self.init_policy = torch.tensor([[0.000, -1.000],
                                         [0.000, -2.000]], dtype=torch.float64)  # 1e-0 optimal
        self.K0 = torch.mm(self.init_policy, self.C)

    def random_initialize(self):

        distribution = MultivariateNormal(torch.zeros(self.num_state), torch.eye(self.num_state))
        state = distribution.sample(sample_shape=torch.Size([self.num_agent])).double()  # torch.Size([1, 4])

        return state

    def policy_noise(self):

        theta = 2 * math.pi * torch.rand(1, )
        phi = 2 * math.pi * torch.rand(1, )
        U = torch.zeros_like(self.init_policy, dtype=torch.float64)

        U[0, 0] = torch.cos(theta) * torch.cos(phi)
        U[0, 1] = torch.cos(theta) * torch.sin(phi)
        U[1, 0] = torch.sin(theta) * torch.cos(phi)
        U[1, 1] = torch.sin(theta) * torch.sin(phi)

        return U

    def rollout_n_step(self, initial_state, F_hat):

        accumulated_cost = torch.zeros([self.num_agent, 1], dtype=torch.float64)
        accumulated_corr = torch.zeros([self.num_agent, self.num_observation, self.num_observation], dtype=torch.float64)

        state = initial_state  # torch.Size([batch_size, 4])
        for i in range(self.num_step):
            observation = torch.mm(state, self.C.t())    # torch.Size([batch_size, 2])
            control = -torch.mm(observation, F_hat.t())  # torch.Size([batch_size, 2])
            state_next = torch.mm(state, self.A.t()) + torch.mm(control, self.B.t())  # torch.Size([batch_size, 4])

            cost = self.cost_func(state, control)
            correlation = torch.bmm(observation[:, :, np.newaxis], observation[:, np.newaxis, :])

            accumulated_cost += cost         # torch.Size([batch_size, 1])
            accumulated_corr += correlation  # torch.Size([batch_size, 2, 2])
            state = state_next

        return accumulated_cost, accumulated_corr

    def cost_func(self, state, control):

        cost = torch.bmm(torch.mm(state, self.Q)[:, np.newaxis, :], state[:, :, np.newaxis]).squeeze(-1) \
               + torch.bmm(torch.mm(control, self.R)[:, np.newaxis, :], control[:, :, np.newaxis]).squeeze(-1)

        return cost

    @staticmethod
    def dlyap(A, Q):
        """
        A X A ^ T - X + Q = 0
        X = dlyap(A, Q)
        """
        assert max(torch.abs(torch.linalg.eigvals(A))).item() < 1.0, 'unstable closed-loop dynamic system'
        ite = 0
        max_ite = 1000
        eps = 1e-8
        X_old = torch.ones_like(A, dtype=torch.float64)
        X_new = torch.zeros_like(A, dtype=torch.float64)
        while ite < max_ite and torch.norm(X_new - X_old).item() > eps:
            ite = ite + 1
            X_old = X_new
            X_new = torch.mm(torch.mm(A, X_old), A.t()) + Q
        # print(f'ite = {ite}, log dlyap error = {math.log10(torch.norm(X_new - X_old).item()):.2f}')
        return X_new

    def lyap(self, K):
        """
        (A - B K)^T P_K (A - B K) - P_K + Q + K^T R K = 0, or
        (A - B F C)^T P_K (A - B F C) - P_K + Q + C^T F^T R F C = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution of Lyapunov Equation
        """
        Ar = self.A - torch.mm(self.B, K)
        P_K = self.dlyap(Ar.t(), self.Q + torch.mm(torch.mm(K.t(), self.R), K))
        return P_K

    def obj_func(self, K):
        """
        J(K) = Tr(P_K X_0) = Tr(P_K Sigma)
        :param K: F * C, where F is the output feedback gain
        :return: the objective function of F * C
        """
        P_K = self.lyap(K)
        J_K = torch.mm(P_K, self.Sigma).trace()
        return J_K

    def state_correlation(self, K):
        """
        (A - B K) Sigma_K (A - B K)^T - Sigma_K + I = 0, or
        (A - B F C) Sigma_K (A - B F C)^T - Sigma_K + I = 0
        :param K: F * C, where F is the output feedback gain
        :return: the solution
        """
        Ar = self.A - torch.mm(self.B, K)
        Sigma_K = self.dlyap(Ar, torch.eye(self.num_state, dtype=torch.float64))
        return Sigma_K


if __name__ == '__main__':
    pass
