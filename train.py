from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import xlwt


# figure size
figure_size = 3.6, 2.7  # PPT
# font size
size_legend = 9
size_label = 12
size_title = 14
# line thickness
size_line = 2.0   # continuous line
size_alpha = 0.3  # transparency
# dpi
size_dpi = 300
# font name
name_TNR = 'Times New Roman'
# legend font
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': size_legend,
               }
# label font
font_label = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': size_label,
              }


def mjrFormatter(x, pos):
    return "$10^{{{0}}}$".format(int(x))


class Train:

    def __init__(self, env, **kwargs):

        super(Train, self).__init__()

        self.num_method = kwargs['num_method']
        self.max_run = kwargs['max_run']
        self.method_name = kwargs['method_name']
        self.method_color = kwargs['method_color']

        self.env = env
        self.num_state = self.env.num_state
        self.num_observation = self.env.num_observation
        self.num_control = self.env.num_control
        self.opt_policy = self.env.opt_policy

        self.num_iteration = kwargs['num_iteration']

        self.num_agent = kwargs['num_agent']
        self.num_policy = kwargs['num_policy']
        self.num_step = kwargs['num_step']
        self.num_record = kwargs['num_record']

        self.policy_index = np.ones([int(self.num_iteration / self.num_record) + 1, 1], dtype="int32")
        self.policy_history = np.ones([self.num_method, self.max_run, 
                                       int(self.num_iteration / self.num_record) + 1, 
                                       self.num_control, self.num_observation], dtype="float64")
        self.policy_accuracy = np.ones([self.num_method, self.max_run, 
                                        int(self.num_iteration / self.num_record) + 1], dtype="float64")
        self.cost_history = np.ones([self.num_method, self.max_run,
                                     int(self.num_iteration / self.num_record) + 1], dtype="float64")
        self.cost_accuracy = np.ones([self.num_method, self.max_run,
                                      int(self.num_iteration / self.num_record) + 1], dtype="float64")
        self.gradient_norm = np.ones([self.num_method, self.max_run,
                                      int(self.num_iteration / self.num_record)], dtype="float64")

        # plot parameters
        self.figure_size = figure_size

    def equivalent_of_lqr(self, Ki):
        A_c = self.env.A - torch.mm(self.env.B, Ki)
        P = self.env.dlyap(A_c.t(), self.env.Q + torch.mm(torch.mm(Ki.t(), self.env.R), Ki))
        Y = self.env.dlyap(A_c, self.env.X)
        yc_minus = torch.mm(torch.mm(self.env.V2,
                                     torch.inverse(torch.mm(torch.mm(self.env.V2.t(), torch.inverse(Y)), self.env.V2))),
                            torch.mm(self.env.V2.t(), torch.inverse(Y)))
        E = torch.mm(torch.mm(torch.inverse(self.env.R + torch.mm(torch.mm(self.env.B.t(), P), self.env.B)),
                              torch.mm(torch.mm(self.env.B.t(), P), self.env.A)),
                     self.env.I_num_state - yc_minus) - Ki
        return E

    def gradient_descent(self, F):
        P_K = self.env.lyap(torch.mm(F, self.env.C))
        Sigma_K = self.env.state_correlation(torch.mm(F, self.env.C))
        E_K = torch.mm(torch.mm(self.env.R + torch.mm(torch.mm(self.env.B.t(), P_K), self.env.B), F), self.env.C) \
              - torch.mm(torch.mm(self.env.B.t(), P_K), self.env.A)
        Delta_J_K = 2 * torch.mm(torch.mm(E_K, Sigma_K), self.env.C.t())
        return Delta_J_K

    def natural_policy_gradient(self, F):
        Delta_J_K = self.gradient_descent(F)
        Sigma_K = self.env.state_correlation(torch.mm(F, self.env.C))
        L = torch.mm(torch.mm(self.env.C, Sigma_K), self.env.C.t())
        Delta_NA = torch.mm(Delta_J_K, torch.inverse(L))
        return Delta_NA

    def gauss_newton_policy_gradient(self, F):
        Delta_NA = self.natural_policy_gradient(F)
        P_K = self.env.lyap(torch.mm(F, self.env.C))
        M = self.env.R + torch.mm(torch.mm(self.env.B.t(), P_K), self.env.B)
        Delta_GN = torch.mm(torch.inverse(M), Delta_NA)
        return Delta_GN

    def estimated_gradient_descent(self, F):
        gradient = torch.zeros([self.num_policy, self.num_control, self.num_observation], dtype=torch.float64)
        for i in range(self.num_policy):

            initial_state_i = self.env.random_initialize()  # sample from a multivariate normal distribution
            U = self.env.policy_noise()
            left_cost_i, _ = self.env.rollout_n_step(initial_state_i, F + self.env.norm_policy_noise * U)
            right_cost_i, _ = self.env.rollout_n_step(initial_state_i, F)
            diff_cost_i = (left_cost_i - right_cost_i) / self.env.norm_policy_noise  # torch.Size([batch_size, 1])
            gradient[i, :, :] = torch.mean(diff_cost_i, dim=0).squeeze() * U

        estimated_Delta_J_K = torch.mean(gradient, dim=0)

        return estimated_Delta_J_K

    def estimated_natural_policy_gradient(self, F):
        accumulated_corr = torch.zeros([self.num_policy, self.num_observation, self.num_observation], dtype=torch.float64)
        gradient = torch.zeros([self.num_policy, self.num_control, self.num_observation], dtype=torch.float64)
        for i in range(self.num_policy):

            initial_state_i = self.env.random_initialize()  # sample from a multivariate normal distribution
            U = self.env.policy_noise()
            left_cost_i, _ = self.env.rollout_n_step(initial_state_i, F + self.env.norm_policy_noise * U)
            right_cost_i, accumulated_corr_i = self.env.rollout_n_step(initial_state_i, F)
            accumulated_corr[i, :, :] = torch.mean(accumulated_corr_i, dim=0)  # torch.Size([batch_size, 2, 2])

            diff_cost_i = (left_cost_i - right_cost_i) / self.env.norm_policy_noise  # torch.Size([batch_size, 1])
            gradient[i, :, :] = torch.mean(diff_cost_i, dim=0).squeeze() * U

        estimated_Delta_J_K = torch.mean(gradient, dim=0)
        L_K = torch.mean(accumulated_corr, dim=0)
        estimated_Delta_NA = torch.mm(estimated_Delta_J_K, torch.inverse(L_K))

        return estimated_Delta_NA

    def save_policy(self, save_file):
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet_4 = book.add_sheet('policy', cell_overwrite_ok=True)

        for m in range(self.num_method):
            for r in range(self.max_run):
                for i in range(int(self.num_iteration / self.num_record) + 1):
                    sheet_4.write(i, 5 * m + 10 * r, self.policy_index[i, 0].tolist())
                    for j in range(self.num_control):
                        for k in range(self.num_observation):
                            sheet_4.write(i, 5 * m + 10 * r + j * self.num_control + k + 1,
                                          self.policy_history[m, r, i, j, k].tolist())
                    sheet_4.write(i, 5 * m + 10 * r + self.num_control * self.num_observation + 1,
                                  self.policy_accuracy[m, r, i].tolist())

        book.save(save_file + 'policy_{:d}'.format(self.num_iteration) + '.xls')

    def save_gradient_norm(self, save_file):
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet_5 = book.add_sheet('gradient_norm', cell_overwrite_ok=True)

        for m in range(self.num_method):
            for r in range(self.max_run):
                for i in range(int(self.num_iteration / self.num_record)):
                    sheet_5.write(i, m + 10 * r, self.gradient_norm[m, r, i].tolist())

        book.save(save_file + 'gradient_norm_{:d}'.format(self.num_iteration) + '.xls')

    def plot_policy(self, save_file, ylim, loc):
        fig1 = plt.figure(1, figsize=self.figure_size)
        ax11 = fig1.add_subplot(111)
        if self.num_method <= 4:
            for m in range(self.num_method):
                plt.plot(self.policy_index, self.policy_accuracy[m, 0, :],
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
        else:
            for m in range(4):
                plt.plot(self.policy_index, self.policy_accuracy[m, 0, :],
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
            for m in range(4, self.num_method):
                plt.fill_between(self.policy_index[:, 0].tolist(),
                                 np.min(self.policy_accuracy[m, :, :], 0).tolist(),
                                 np.max(self.policy_accuracy[m, :, :], 0).tolist(),
                                 facecolor=self.method_color[m], alpha=size_alpha)
                plt.plot(self.policy_index, np.mean(self.policy_accuracy[m, :, :], 0),
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
        # legend
        plt.legend(loc=loc, prop=font_legend)
        # coordinate
        plt.xlim((0, self.num_iteration))
        plt.ylim(ylim)
        plt.gca().set_yscale('log')
        plt.tick_params(axis='both', labelsize=size_label)
        labels = ax11.get_xticklabels() + ax11.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # label
        plt.xlabel('Iteration', font_label)
        plt.ylabel('Relative Policy Error', font_label)
        # save
        plt.savefig(save_file + 'policy_error_{:d}'.format(self.num_iteration) + '.svg',
                    format='svg', dpi=size_dpi, bbox_inches='tight')

        return fig1

    def plot_cost(self, save_file, ylim, loc):
        fig2 = plt.figure(2, figsize=self.figure_size)
        ax21 = fig2.add_subplot(111)
        if self.num_method <= 4:
            for m in range(self.num_method):
                plt.plot(self.policy_index, self.cost_accuracy[m, 0, :],
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
        else:
            for m in range(4):
                plt.plot(self.policy_index, self.cost_accuracy[m, 0, :],
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
            for m in range(4, self.num_method):
                plt.fill_between(self.policy_index[:, 0].tolist(),
                                 np.min(self.cost_accuracy[m, :, :], 0).tolist(),
                                 np.max(self.cost_accuracy[m, :, :], 0).tolist(),
                                 facecolor=self.method_color[m], alpha=size_alpha)
                plt.plot(self.policy_index, np.mean(self.cost_accuracy[m, :, :], 0),
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
        # legend
        plt.legend(loc=loc, prop=font_legend)
        # coordinate
        plt.xlim((0, self.num_iteration))
        plt.ylim(ylim)
        plt.gca().set_yscale('log')
        plt.tick_params(axis='both', labelsize=size_label)
        labels = ax21.get_xticklabels() + ax21.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # label
        plt.xlabel('Iteration', font_label)
        plt.ylabel('Relative Cost Error', font_label)
        # save
        plt.savefig(save_file + 'cost_error_{:d}'.format(self.num_iteration) + '.svg',
                    format='svg', dpi=size_dpi, bbox_inches='tight')

        return fig2

    def plot_gradient_norm(self, save_file, ylim, loc):
        fig3 = plt.figure(3, figsize=self.figure_size)
        ax31 = fig3.add_subplot(111)
        index = np.arange(int(self.num_iteration / self.num_record))
        if self.num_method <= 4:
            for m in range(self.num_method):
                plt.plot(index, self.gradient_norm[m, 0, :],
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
        else:
            for m in range(4):
                plt.plot(index, self.gradient_norm[m, 0, :],
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
            for m in range(4, self.num_method):
                plt.fill_between(index.tolist(),
                                 np.min(self.gradient_norm[m, :, :], 0).tolist(),
                                 np.max(self.gradient_norm[m, :, :], 0).tolist(),
                                 facecolor=self.method_color[m], alpha=size_alpha)
                plt.plot(index, np.mean(self.gradient_norm[m, :, :], 0),
                         c=self.method_color[m], linewidth=size_line, label=self.method_name[m])
        # legend
        plt.legend(loc=loc, prop=font_legend)
        # coordinate
        plt.xlim((0, self.num_iteration))
        plt.ylim(ylim)
        plt.gca().set_yscale('log')
        plt.tick_params(axis='both', labelsize=size_label)
        labels = ax31.get_xticklabels() + ax31.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # label
        plt.xlabel('Iteration', font_label)
        plt.ylabel('Gradient Norm', font_label)
        # save
        plt.savefig(save_file + 'gradient_norm_{:d}'.format(self.num_iteration) + '.svg',
                    format='svg', dpi=size_dpi, bbox_inches='tight')

        return fig3

