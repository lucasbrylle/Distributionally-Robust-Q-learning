
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from irlc.ex09.mdp import MDP
from irlc.ex09.value_iteration import value_iteration
from irlc.ex09.value_iteration import qs_
from irlc.ex09.policy_evaluation import policy_evaluation
from irlc.ex09.policy_iteration import policy_iteration


# Made by Tuhe
# Minor changes by s203832
# THIS IS THE ONE TO USE

def exact_solution(mdp, gamma=0.9):
    a, v = value_iteration(mdp, gamma=gamma)

    qs = {}
    for s in mdp.nonterminal_states:
        qs[s] = list(qs_(mdp, s=s, gamma=gamma, v=v).values())

    return qs, {s: np.argmax(q) for s, q in qs.items()}

"""The following MDP-definition uses the MDP-framework from Tue Herlaus course 02465 - "Introduction to reinforcement learning and control theory"""

class SupplyChainModel(MDP):

    def __init__(self, n=10, b=0, m=0):
        self._states = range(n+1)
        self._nonterminal_states = range(n+1)
        self._terminal_states = None
        self.b = b
        self.m = m

        self.demand_dist = {}

        """Demand distribution for the given parameters b and m"""
        for d in range(n + 1):
            if (d == m) or (d == m + 1):
                pd = (b + 1) / (n + 1)
            else:
                pd = (n - 1 - (2 * b)) / ((n ** 2) - 1)
            self.demand_dist[d] = pd

        assert np.abs(sum(self.demand_dist.values()) - 1) < 1e-8
        self.n = n
        self.initial_state = 0

        super().__init__(initial_state=0)

    def is_terminal(self, state):  # !f Return true only if state is terminal.
        """ Implement if the state is terminal (0 or self.goal) """
        return False

    """Action space for a given state"""
    def A(self, s):
        return list(range(self.n + 1 - s))

    """
    Dictionary for s' and reward probabilites
    """
    def Psr(self, s, a):

        # Parameters for cost function
        h = 1
        p = 2
        k = 3

        pp = defaultdict(float)

        if a not in list(range(self.n + 1 - s)):
            ap = self.n - s

        else:
            ap = a

        for d in range(self.n + 1):
            pd = self.demand_dist[d]

            s_next = max(0, s + ap - d)

            h_l = max(0, (s + ap - d))
            p_l = max(0, (d - s - ap))

            ct = k * (ap > 0) + h * h_l + p * p_l

            pp[(s_next, ct)] += pd
        assert np.abs(sum(pp.values()) - 1) < 1e-8
        return pp

    """From 02465 - "Introduction to reinforcement learning and control theory"""
    def p_sa(self, s, a):
        """ Represent p_{s,a} from the paper. I.e. the distribution of p(s' | s,a)"""
        pp = defaultdict(float)
        for (sp, r), p in self.Psr(s, a).items():
            pp[sp] += p
        return pp

    """From 02465 - "Introduction to reinforcement learning and control theory"""
    def get_KL_distance_from(self, p0_sa):
        """
        Let the distribution in this class be p(s', r | s,a). This will compute the KL distance from that distribution to
        q (represented as a dictionary).

        let p0_sa be the original distribution - in our case the uniform non pertubed distribution

        """
        kl = []
        for s in self.nonterminal_states:
            for a in self.A(s):
                q = p0_sa[s, a]
                p = self.p_sa(s, a)
                states = set(p) | set(q)
                kl_ = [p.get(sp, 0) * np.log(p.get(sp, 0) / q.get(sp, 0)) if p.get(sp, 0) > 0 else 0 for sp in states]
                kl.append(sum(kl_))

        delta = max(kl)
        return delta
