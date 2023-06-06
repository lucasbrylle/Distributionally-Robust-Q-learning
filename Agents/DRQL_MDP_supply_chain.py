import multiprocessing
import random
from multiprocessing import Pool, cpu_count
import numpy as np
import decimal
import math
import decimal
from itertools import repeat
from scipy.special import logsumexp, expit
from BachelorDRQL.Environments.supply_chain_env import SupplyChainModel
from irlc.ex09.mdp import rng_from_dict
def f(array, var, delta):
    m_og = min(array)
    asarray = np.asarray(array)
    z = (asarray - m_og) / var
    exp_term = [decimal.Decimal(-element).exp() for element in z]
    exp_term = np.asarray([float(element) for element in exp_term])

    v = m_og - var * (np.log(np.mean(exp_term))) - (var * delta)
    return v
def f_diff(array, var, delta):
    m_diff = min(array)
    asarray = np.asarray(array)
    z = (asarray - m_diff) / var
    exp_term = [decimal.Decimal(-element).exp() for element in z]
    exp_term = np.asarray([float(element) for element in exp_term])

    #v = m_diff - delta - np.log(np.mean(exp_term)) - m_diff - (np.sum(z * exp_term) / np.sum(exp_term))
    v = m_diff - delta - np.log(np.mean(exp_term)) - (np.sum(z * exp_term) / np.sum(exp_term))
    return v
"""
def f(array, var, delta):
    m_og = min(array)
    asarray = np.asarray(array)
    z = (asarray - m_og) / var
    exp_term = [decimal.Decimal(-element).exp() for element in z]
    exp_term = np.asarray([float(element) for element in exp_term])

    v = m_og - var * (np.log(np.mean(exp_term))) - (var * delta)
    return v
def f_diff(array, var, delta):
    
    #v = m_diff - delta - np.log(np.mean(np.exp((-((array) - m_diff)) / var))) - m_diff - ((np.sum(((array - m_diff) / var) * (np.exp(((array) - m_diff) / var)))) / (np.sum(np.exp(-((array) - m_diff) / var))))

    #arrayD = np.asarray([decimal.Decimal(el) for el in x], dtype=object)

    k = len(array)
    array = np.asarray(array, dtype=np.float64)
    x = array/var
    x_ = -array/var

    z = []
    for elem in x_:
        if elem == 0:
            z.append(1)
        elif elem < 0:
            z.append(np.exp(elem)/(1+np.exp(elem)))
        else:
            z.append(1/(1+np.exp(-elem)))
    v = - delta - np.log((1/k) * np.sum(z)) - ((np.sum(x * z)) / (np.sum(z)))
    return v
 
    m_diff = min(array)
    asarray = np.asarray(array)
    z = (asarray - m_diff) / var
    exp_term = [decimal.Decimal(-element).exp() for element in z]
    exp_term = np.asarray([float(element) for element in exp_term])

    v = m_diff - delta - np.log(np.mean(exp_term)) - m_diff - (np.sum(z * exp_term) / np.sum(exp_term))
    return v
"""
def init_Q(n_s, n_a):
    Q = {}
    for s in range(n_s):
        Q[s] = {}
        for a in range(n_a):
            Q[s][a] = 0
    return Q
def stopping_time(epsilon):
    n = np.random.geometric(p=epsilon, size=1)[0]
    return n
def get_Q_max(Q, states, delta):
    Q_array = []
    for s in states:
        (actions, Qa) = zip(*Q[s].items())
        Qa_ = np.argmax(np.asarray(Qa) + np.random.rand(len(Qa)) * 1e-8)
        Q_array.append(Q[s][Qa_])

    return Q_array
def generate_samples(N, state, action, mdp):
    s_next = []
    rewards = []
    ps = mdp.Psr(state, action)

    for it in range(N):
        s_, reward = rng_from_dict(ps)
        s_next.append(s_)
        rewards.append(-reward)

    return s_next, rewards
def bisection_method(array_bi, delta, a, b, var, tol):
    out_diff = f_diff(array_bi, var, delta)

    if np.abs(out_diff) < tol:
        return var

    if out_diff < 0:
        return bisection_method(array_bi, delta, a, var, (a + var) / 2, tol)
    if out_diff > 0:
        return bisection_method(array_bi, delta, var, b, (b + var) / 2, tol)
def find_sup(array_sup, delta):
    # if np.all(np.array(array_sup[0] == np.array(array_sup))):
    #    return array_sup[0]

    if f_diff(array_sup, 0.001, delta) < 0:
        sup = f(array_sup, 0.001, delta)
    else:
        var = bisection_method(array_bi=array_sup, delta=delta, a=0.001, b=10000, var=2, tol=0.01)
        sup = f(array_sup, var, delta)
    if sup < 0:
        sup = 0
    return sup
    """
    if np.all(np.array(array_sup[0] == np.array(array_sup))):
        return array_sup[0]
    if np.sign(f_diff(array_sup, 0.001, delta)) == np.sign(f_diff(array_sup, 10000, delta)):
        sup = f(array_sup, 0.001, delta)
    elif f_diff(array_sup, 0.001,  delta) < 0:
        sup = f(array_sup, 0.001, delta)
    elif np.log(len(array_sup) / array_sup.count(min(array_sup))) <= delta:
        sup = f(array_sup, 0.001, delta)
    else:
        var = bisection_method(array_sup, delta, 0.001, 1000, 1, 0.001)
        sup = f(array_sup, var, delta)

    """
def Delta(d, delta):
    d_2nd = d[::2]
    d_2nd_m1 = d[1::2]

    l1 = find_sup(d, delta)
    l2 = find_sup(d_2nd, delta)
    l3 = find_sup(d_2nd_m1, delta)

    l = l1 - 0.5 * l2 - 0.5 * l3

    return l
def DRQL_math(mdp, Q, gamma, epsilon, tolerance, delta):
    t = 1
    # n_array = []
    samples = 0
    converged = False
    while not converged:
        delta_convergence = -np.inf
        count = 0
        for state in mdp.states:
            for action in mdp.A(0):
                N_samples = stopping_time(epsilon)
                #Trukket en fra N_samples...
                samples += 2 ** (N_samples)
                p_n = epsilon * (1 - epsilon) ** (N_samples)

                s_, r = generate_samples(2**(N_samples), state, action, mdp)
                Q_max = get_Q_max(Q, s_, delta)

                delta_r = Delta(r, delta)
                delta_Q = Delta(Q_max, delta)

                r_sample = delta_r / p_n
                Q_sample = delta_Q / p_n

                R_rob = r[0] + r_sample
                T_rob = Q_max[0] + Q_sample

                T_epsilon = R_rob + gamma * T_rob

                alpha_t = 1 / (1 + (1 - gamma) * (t - 1))
                # Alm. Q-læring: T_epsilon = R + gamma max_b Q[s´, b]
                Q_update = (1 - alpha_t) * Q[state][action] + alpha_t * T_epsilon

                diff_not_abs = Q_update - Q[state][action]
                diff = abs(Q_update - Q[state][action])
                Q[state][action] = Q_update
                # breakpoint()
                if diff > delta_convergence:
                    delta_convergence = diff
                count += 1
        if t % 500 == 0:
            print("MDP M = ", mdp.m)
            print(delta_convergence, t)
            print(Q)

        if delta_convergence < tolerance:
            converged = True

        t += 1
    return Q, t, samples


Q_res = []
t_res = []
sample_res = []


def mainLoop(b, m, n=10, delta=1):
    print("Started - M = ", m, flush=True)
    mdp = SupplyChainModel(n=n, b=b, m=m)
    Q = init_Q(len(mdp.states), len(mdp.A(0)))
    Q, t_it, sample_it = DRQL_math(mdp, Q, gamma=0.9, epsilon=0.5, tolerance=0.05, delta=delta)
    return Q, t_it, sample_it, m


if __name__ == "__main__":
    decimal.getcontext().prec = 5
    """
    for m in range(10):
        mdp = SupplyChainModel(n=10, b=1, m=5)
        Q = init_Q(len(mdp.states), len(mdp.A(0)))
        Q, t_it, sample_it = DRQL_math(mdp, Q, gamma=0.9, epsilon=0.5, tolerance=0.05, delta=1)
        Q_res.append(Q)
        t_res.append(t_it)
        sample_res.append(sample_it)
    print(Q_res)
    print(t_res)
    print(sample_res)
    """
    m_args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    res_total1 = []
    res_total2 = []


    for kolort in range(1):
        with Pool(processes=cpu_count()) as pool0:
            res = pool0.starmap(mainLoop, zip(repeat(1), m_args, repeat(10), repeat(0)))
        print(res)
        res_total1.append(res)
    print(res_total1)

    for kolort in range(1):
        with Pool(processes=cpu_count()) as pool1:
            res = pool1.starmap(mainLoop, zip(repeat(1), m_args, repeat(10), repeat(1)))
        print(res)
        res_total2.append(res)

    print(res_total1)
    print(res_total2)
