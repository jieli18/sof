from __future__ import print_function
from dynamics import DynamicsConfig, Circuit
from train import Train
from tensorboardX import SummaryWriter

from utils import change_type
import argparse
import copy
import json
import torch
import math
import datetime
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    # Parameters Setup
    parser = argparse.ArgumentParser()

    # Key Parameters for users
    parser.add_argument('--list_method', type=list, default=[0, 1, 2, 3, 4], help='6')
    parser.add_argument('--max_run', type=int, default=10, help='10')
    parser.add_argument('--num_run', type=list,
                        default=[1, 1, 1, parser.parse_args().max_run, parser.parse_args().max_run, 1])
    parser.add_argument('--method_name', type=dict,
                        default={0: 'Vanilla Gradient', 1: 'Natural Gradient', 2: 'Gauss-Newton',
                                 3: 'Model-free Vanilla', 4: 'Model-free Natural', 5: 'method in [46]'})
    parser.add_argument('--method_color', type=dict,
                        default={0: '#4169E1', 1: '#9932CC', 2: '#FF8000', 3: 'r', 4: '#32CD32', 5: 'black'})

    # 1. Parameters for environment
    parser.add_argument('--env_name', type=str, default='Circuit')
    parser.add_argument('--num_state', type=int, default=4, help='')
    parser.add_argument('--num_observation', type=int, default=2, help='')
    parser.add_argument('--num_control', type=int, default=2, help='')
    parser.add_argument('--random_init_policy', type=bool, default=False)

    # 2. Parameters for algorithm
    parser.add_argument('--ite', type=int, default=1)
    parser.add_argument('--num_iteration', type=int, default=300, help='300')
    parser.add_argument('--policy_learning_rate', type=list, default=[10, 50, 1e-1, 36, 175, 1e-1])

    # 3. Parameters for trainer
    # Parameters for sampler
    parser.add_argument('--num_agent', type=int, default=128, help='128')
    parser.add_argument('--num_policy', type=int, default=128, help='128')
    parser.add_argument('--num_step', type=int, default=50, help='50')
    parser.add_argument('--norm_policy_noise', type=float, default=1e-3)
    # Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--num_record', type=int, default=1, help='for polynomial approximators')
    parser.add_argument('--num_print', type=int, default=10)

    # get parameter dict
    args = vars(parser.parse_args())
    DynamicsConfig.num_agent = args['num_agent']
    DynamicsConfig.num_policy = args['num_policy']
    DynamicsConfig.num_step = args['num_step']
    DynamicsConfig.norm_policy_noise = args['norm_policy_noise']
    DynamicsConfig.random_init_policy = args['random_init_policy']

    env = Circuit()
    args['num_state'] = env.num_state
    args['num_observation'] = env.num_observation
    args['num_control'] = env.num_control

    lr_policy = args['policy_learning_rate']
    scaling_factor = lr_policy[5]

    def clip_gradient(norm_gradient):
        return 5.0 / norm_gradient if norm_gradient > 5.0 else 1.0

    opt_policy = env.opt_policy  # F^*
    norm_opt_policy = torch.norm(opt_policy)  # || F^* ||_Fro

    opt_cost = env.obj_func(torch.mm(opt_policy, env.C))  # J(K^*) = J(F^* * C)
    norm_opt_cost = torch.abs(opt_cost)  # | J(K_0) - J(K^*) | = | J(F_0 * C) - J(F^* * C) |

    train = Train(env, **args)
    train.figure_size = 3.6, 2.7  # PPT

    save_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
    results_file = './results/' + save_time
    save_file = results_file + '/'
    writer = SummaryWriter(results_file)
    args['save_folder'] = save_file

    for m in args['list_method']:
        for r in range(args['num_run'][m]):
            args['ite'] = 1
            train.policy_index[0] = 0
            policy = env.init_policy()
            Ki = torch.mm(policy, env.C)  # K = F * C, u = - F * y = - F * C * x = - K * x
            init_cost = env.obj_func(Ki)  # J(K_0) = J(F_0 * C)
            cost = init_cost  # J(K_i) = J(F_i * C)

            train.policy_history[m, r, 0, :, :] = policy  # F_i
            error_policy = torch.norm(policy - opt_policy)  # || F_i - F^* ||_Fro
            train.policy_accuracy[m, r, 0] = error_policy / norm_opt_policy

            train.cost_history[m, r, 0] = cost  # J(K_i) = J(F_i * C)
            error_cost = torch.abs(cost - opt_cost)  # | J(K_i) - J(K^*) |
            train.cost_accuracy[m, r, 0] = error_cost / norm_opt_cost

            for i in range(args['num_iteration']):

                if args['method_name'][m] == 'Vanilla Gradient':
                    Delta_J_K = train.gradient_descent(policy)
                    norm_Delta_J_K = torch.norm(Delta_J_K)
                    train.gradient_norm[m, r, i] = norm_Delta_J_K
                    policy = policy - lr_policy[m] * Delta_J_K * clip_gradient(norm_Delta_J_K)
                elif args['method_name'][m] == 'Natural Gradient':
                    Delta_NA = train.natural_policy_gradient(policy)
                    norm_Delta_NA = torch.norm(Delta_NA)
                    train.gradient_norm[m, r, i] = norm_Delta_NA
                    policy = policy - lr_policy[m] * Delta_NA * clip_gradient(norm_Delta_NA)
                elif args['method_name'][m] == 'Gauss-Newton':
                    Delta_GN = train.gauss_newton_policy_gradient(policy)
                    norm_Delta_GN = torch.norm(Delta_GN)
                    train.gradient_norm[m, r, i] = norm_Delta_GN
                    policy = policy - lr_policy[m] * Delta_GN * clip_gradient(norm_Delta_GN)
                elif args['method_name'][m] == 'Model-free Vanilla':
                    estimated_Delta_J_K = train.estimated_gradient_descent(policy)
                    norm_estimated_Delta_J_K = torch.norm(estimated_Delta_J_K)
                    train.gradient_norm[m, r, i] = norm_estimated_Delta_J_K
                    policy = policy - lr_policy[m] * estimated_Delta_J_K * clip_gradient(norm_estimated_Delta_J_K)
                elif args['method_name'][m] == 'Model-free Natural':
                    estimated_Delta_NA = train.estimated_natural_policy_gradient(policy)
                    norm_estimated_Delta_NA = torch.norm(estimated_Delta_NA)
                    train.gradient_norm[m, r, i] = norm_estimated_Delta_NA
                    policy = policy - lr_policy[m] * estimated_Delta_NA * clip_gradient(norm_estimated_Delta_NA)
                elif args['method_name'][m] == 'method in [46]':
                    E = train.equivalent_of_lqr(Ki)
                    norm_E = torch.norm(E)
                    train.gradient_norm[m, r, i] = norm_E
                    Ki = Ki + scaling_factor * E * clip_gradient(norm_E)
                    policy = torch.mm(Ki, env.C_pseudo_inv)
                else:
                    print('method name error!')
                    exit()

                # # determine whether the output feedback is stable
                # print(f'ite: {i}, maximum eigenvalue of A - B * Fi * C: '
                #       f'{max(torch.abs(torch.linalg.eigvals(env.A - env.B @ policy @ env.C))).item():.4f}')

                if args['method_name'][m] == 'method in [46]':
                    cost = env.obj_func(Ki)  # J(K_i) = J(F_i * C)
                else:
                    cost = env.obj_func(torch.mm(policy, env.C))  # J(K_i) = J(F_i * C)

                train.policy_index[i + 1] = i + 1

                train.policy_history[m, r, i + 1, :, :] = policy  # F_i
                error_policy_pre = torch.norm(policy - opt_policy)  # || F_i - F^* ||_Fro
                if error_policy_pre > 1e-15:
                    error_policy = error_policy_pre
                train.policy_accuracy[m, r, i + 1] = error_policy / norm_opt_policy

                train.cost_history[m, r, i + 1] = cost  # J(K_i) = J(F_i * C)
                error_cost_pre = torch.abs(cost - opt_cost)  # | J(K_i) - J(K^*) |
                if error_cost_pre > 1e-15:
                    error_cost = error_cost_pre
                train.cost_accuracy[m, r, i + 1] = error_cost / norm_opt_cost

                # print training process
                if i % args['num_print'] == 0:
                    print(f"method{m}: {args['method_name'][m]}, run: {r}, ite: {i}, "
                          f"log error of policy = {math.log10(error_policy / norm_opt_policy):.2f}, "
                          f"log error of cost = {math.log10(error_cost / norm_opt_cost):.2f}")

                args['ite'] = args['ite'] + 1

    print('Finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with open(save_file + '/config.json', 'w', encoding='utf-8') as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)
    train.save_policy(save_file)
    train.save_gradient_norm(save_file)
    fig1 = train.plot_policy(save_file, ylim=(0.5e-3, 2e0), loc='upper right')
    fig2 = train.plot_cost(save_file, ylim=(0.9e-7, 2e0), loc='upper right')
    fig3 = train.plot_gradient_norm(save_file, ylim=(0.9e-5, 2e2), loc='upper right')
    for method_id in range(len(args['list_method'])):
        if args['num_run'][method_id] > 1:
            train.plot(save_file, method_id)
    plt.show()

    writer.add_figure('policy error', fig1, close=False)
    writer.add_figure('cost error', fig2, close=False)
    writer.add_figure('gradient norm', fig3, close=False)


if __name__ == '__main__':
    main()
