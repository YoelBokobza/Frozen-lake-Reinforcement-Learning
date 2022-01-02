# -*- coding: utf-8 -*-
"""
Created on Sat May 29 11:05:11 2021

@author: yoell
"""

from env import World
import numpy as np
import time
import cProfile
from matplotlib import pyplot as plt
import time
start = time.time()
def scan_best_alpha(sweeps, algofunc, title_name, gamma, value):
    alpha_vec = [0.001, 0.0025, 0.005, 0.01]
    num_episodes = int(1e6)
    results = np.zeros((len(alpha_vec), sweeps, num_episodes))
    for sweep in range(sweeps):
        print(f'Sweep Index {sweep}')
        for idx, alpha in enumerate(alpha_vec):
            print(f'Alpha Value {alpha}')
            policy, Q, V, RMS = algofunc(num_episodes, alpha, gamma=gamma,optimal_V=value)
            results[idx,sweep,:] = RMS
    legend_vec = []
    plt.figure()
    plt.grid()
    episode_vec = np.arange(num_episodes)

    for idx,alpha in enumerate(alpha_vec):
        legend_vec.append(r'$\alpha$ = ' + str(alpha))
        plt.plot(episode_vec, np.mean(results[idx], axis=0))

    # plt.plot(alpha_vec, results)
    plt.xlabel('Episode')
    plt.ylabel('RMS Error')
    plt.legend(legend_vec)
    plt.title(r'$\alpha$ values sweep ' + title_name)

    plt.savefig(title_name, dpi=500)
    plt.show()

def question_1():
    
    env = World()
    
    theta = 1e-12
    gamma = 0.9
    policy, value, action_value = env.policy_iteration(theta,gamma)

    env.plot_policy(policy, title='Optimal Policy')
    env.plot_value(value, title='Optimal State Value')
    env.plot_qvalue(action_value, title='Optimal State-Action Value')

def question_2():

    env = World()
    num_episodes = int(1e6)

    alpha = 0.001
    gamma = 0.9


    # theta = 1e-12
    # policy, value, action_value = env.policy_iteration(theta, gamma)

    # SWEEPS = 10
    # scan_best_alpha(SWEEPS, env.mc_control, 'Monte-Carlo', gamma, value)
    policy, Q, V, RMS = env.mc_control(num_episodes, alpha, gamma=gamma)

    env.plot_policy(policy, title='MC-control Policy')
    env.plot_value(V, title='MC-control State Value')
    env.plot_qvalue(Q, title='MC-control State-Action Value')


def question_3():
    
    env = World()
    num_episodes = int(1e5)
    
    alpha = 0.001
    gamma = 0.9

    # theta = 1e-12
    # policy, value, action_value = env.policy_iteration(theta, gamma)

    # SWEEPS = 10
    # scan_best_alpha(SWEEPS, env.sarsa, 'SARSA', gamma, value)

    
    policy, Q, V, RMS = env.sarsa(num_episodes, alpha, gamma=gamma)

    env.plot_policy(policy, title='SARSA Policy')
    env.plot_value(V, title='SARSA State Value')
    env.plot_qvalue(Q, title='SARSA State-Action Value')

def question_4():

    env = World()
    num_episodes = int(1e6)
    
    alpha = 0.001
    gamma = 0.9


    # theta = 1e-12
    # policy, value, action_value = env.policy_iteration(theta, gamma)

    # SWEEPS = 10
    # scan_best_alpha(SWEEPS, env.q_learning, 'Q-Learning', gamma, value)
    policy, Q, V, RMS = env.q_learning(num_episodes,alpha, gamma=gamma)

    env.plot_policy(policy, title='Q-Learning control Policy')
    env.plot_value(V, title='Q-Learning State Value')
    env.plot_qvalue(Q, title='Q-Learning State-Action Value')

def algoritms_analysis():
    env = World()

    num_episodes = int(1e6)
    gamma = 0.9
    theta = 0.000000000001
    policy, value, action_value = env.policy_iteration(theta, gamma)


    BEST_ALPHA_Q_LEARNING = 0.001
    BEST_ALPHA_MC = 0.001
    BEST_ALPHA_SARSA = 0.001

    MONTE_CARLO_SIZE = 1
    NUMBER_OF_ALGORITHMS = 3
    results = np.zeros((NUMBER_OF_ALGORITHMS, MONTE_CARLO_SIZE, num_episodes))
    loop_dict = {'MC': {'alpha': BEST_ALPHA_MC, 'func': env.mc_control}, 'SARSA': {'alpha': BEST_ALPHA_SARSA, 'func': env.sarsa}, 'Q Learning': {'alpha': BEST_ALPHA_Q_LEARNING, 'func': env.q_learning}}

    for idx,key in enumerate(loop_dict):
        print(f'Starting {key} Analysis')
        print(idx)
        for monte_carlo_idx in range(MONTE_CARLO_SIZE):
            print(f'MC IDX - {monte_carlo_idx}')
            policy, Q, V, RMS = loop_dict[key]['func'](num_episodes, alpha=loop_dict[key]['alpha'], gamma=gamma, optimal_V= value)
            results[idx,monte_carlo_idx, :] = RMS

    plt.figure()
    legend_vec = []
    episode_vec = np.arange(num_episodes)
    for idx,key in enumerate(loop_dict):
        legend_vec.append(key)
        if MONTE_CARLO_SIZE != 1:
            plt.plot(episode_vec, np.mean(results[idx], axis=0))
        else:
            plt.plot(episode_vec, np.squeeze(results[idx]))
    plt.grid()
    plt.legend(legend_vec)
    plt.xlabel('Episode')
    plt.ylabel('RMS Error')
    plt.title('Algorithms Performance Analysis')

    plt.savefig('Algorithms Performance Analysis', dpi=500)

if __name__ == "__main__":
    #algoritms_analysis()
    #question_1()
    #question_2()
    question_3()
    #question_4()
    end = time.time()
    print(end - start)

































