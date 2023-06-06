import random
from multiprocessing import Pool, cpu_count
import numpy as np
import decimal
from itertools import repeat
from irlc.ex09.rl_agent import TabularAgent
from BachelorDRQL.Environments.cliffwalking_online_env import CliffWalkingEnvironment


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
        Q_array.append(Q[s][actions[Qa_]])
    return Q_array
def generate_samples(N, state, action,env):
    s_next = []
    rewards = []

    for it in range(N):
        s_, r, done, _, info_s = env.sample_rewards(state, action)
        s_next.append(s_)
        rewards.append(r)

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

class DRQL_agent(TabularAgent):
    def __init__(self, env, gamma=0.9, epsilon_dist=0.5, epsilon_disc=0.1, delta=1, t=1):
        self.t = t
        self.epsilon_dist = epsilon_dist
        self.epsilon_disc = epsilon_disc
        self.gamma = gamma
        self.delta = delta
        super().__init__(env, gamma, epsilon_disc)

    def pi(self, s, info=None):
        actions = self.env.A(s)

        return random.choice(actions) if np.random.rand() < self.epsilon else self.Q.get_optimal_action(s, info)

    def DRQL_math(self, state):
        samples = 0
        delta_convergence = 0
        for action in self.env.A(state):
            N_samples = stopping_time(self.epsilon_dist)
            samples += 2 ** (N_samples)
            p_n = self.epsilon_dist * (1 - self.epsilon_dist) ** (N_samples)

            s_, r = generate_samples(2 ** (N_samples), state, action, self.env)
            Q_max = get_Q_max(self.Q.q_, s_, self.delta)

            delta_r = Delta(r, self.delta)
            delta_Q = Delta(Q_max, self.delta)

            r_sample = delta_r / p_n
            Q_sample = delta_Q / p_n

            R_rob = r[0] + r_sample
            T_rob = Q_max[0] + Q_sample

            T_epsilon = R_rob + self.gamma * T_rob

            alpha_t = 1 / (1 + (1 - self.gamma) * (self.t - 1))
            #alpha_t = 0.5
            # Alm. Q-læring: T_epsilon = R + gamma max_b Q[s´, b]
            Q_update = (1 - alpha_t) * self.Q.q_[state][action] + alpha_t * T_epsilon

            diff_not_abs = Q_update - self.Q.q_[state][action]
            diff = abs(Q_update - self.Q.q_[state][action])
            self.Q.q_[state][action] = Q_update
            # breakpoint()
            if diff > delta_convergence:
                delta_convergence = diff
        return delta_convergence, samples

    def train(self, state, done=False, info_s=None, info_sp=None, gamma=0.9, goals=1000, tolerance=0.05):
        converged = False
        samples = 0
        goal_count = 0
        while goals > goal_count:
            # print(self.Q.q_, self.t)
            delta_convergence, sample_it = self.DRQL_math(state)
            samples += sample_it
            a = self.pi(state)
            state, reward, done, _, info_s = self.env.step(a)

            if done:
                goal_count += 0.5
                self.t = goal_count
                self.env.reset()

                if goal_count % 9000 == 0:
                    print("GOAL ", goal_count)
                    print(self.Q.q_)

        return self.t, samples


Q_res = []
t_res = []
sample_res = []
res = []

def MpLoop(step_down, delta=0):
    print("Started  ", step_down, flush=True)
    env_ = CliffWalkingEnvironment(step_down=step_down, render_mode="ansi")
    s0, info = env_.reset()
    agent = DRQL_agent(env_, gamma=0.97, epsilon_dist=0.5, epsilon_disc=0.12, delta=delta, t=1)
    t_it, sample_it = agent.train(s0, goals=20000, tolerance=0.05)
    env_.close()
    print("Finished ", step_down)
    print((agent.Q.q_, sample_it, step_down))
    return agent.Q.to_dict(), t_it, sample_it, step_down

if __name__ == "__main__":
    step_array = [0.025, 0.05, 0.075, 0.1, 0.125, 0.150, 0.175, 0.2]

    res_total0 = []
    for delta0_it in range(1):
        with Pool(processes=cpu_count()) as pool1:
            res = pool1.starmap(MpLoop, zip(step_array, repeat(0)))
        print(res)
        res_total0.append(res)
    print(res_total0)

    res_total1 = []
    for delta1_it in range(1):
        with Pool(processes=cpu_count()) as pool1:
            res = pool1.starmap(MpLoop, zip(step_array, repeat(1)))
        print(res)
        res_total1.append(res)
    print(res_total0)
    print(res_total1)





