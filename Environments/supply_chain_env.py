
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


class SupplyChainModel(MDP):

    def __init__(self, n=10, b=0, m=0):
        self._states = range(n+1)
        self._nonterminal_states = range(n+1)
        self._terminal_states = None
        self.b = b
        self.m = m


        # Demand dist is the distribution used for the given MDP
        self.demand_dist = {}

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


    # Not done - since non-terminal
    def is_terminal(self, state):  # !f Return true only if state is terminal.
        """ Implement if the state is terminal (0 or self.goal) """
        return False

    def A(self, s):
        return list(range(self.n + 1 - s))

    def Psr(self, s, a):
        """
        Old code can be found in junk
        """
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


    def p_sa(self, s, a):
        """ Represent p_{s,a} from the paper. I.e. the distribution of p(s' | s,a)"""
        pp = defaultdict(float)
        for (sp, r), p in self.Psr(s, a).items():
            pp[sp] += p
        return pp

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


if __name__ == "__main__":
    # Creation of MDP
    mdp = SupplyChainModel(b=1, m=0)
    print(mdp.nonterminal_states)
    print("Uniform distribution: ", mdp.demand_dist)
    #q_uniform = {sp: 1/(mdp.n+1) for sp in range(mdp.n+ 1)}
    p0_sa = {}

    for s in range(mdp.n + 1):
        for a in range(mdp.n + 1):
            p0_sa[s,a] = mdp.p_sa(s, a)

    b_array = [1]
    m_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for b_val in b_array:
        m_worst_case = [-math.inf for i in range(10)]
        index = 0

        for m_val in m_array:
            mdp = SupplyChainModel(n = 10, b=b_val, m=m_val)
            pi, v = value_iteration(mdp, gamma=0.90)
            print(v)
            print(pi)
            m_worst_case[m_val] = max(v.values())
            index += 1
        mdp = SupplyChainModel(n=10, b=b_val, m=0)
        pi, v = policy_iteration(mdp, gamma=0.9)
        print(pi)
        print("M_WC", m_worst_case)
        #mDRQL = np.array([-112.58941607745345, -109.76472047492409, -109.15195337360718, -109.86018458914815, -109.41617485748958, -109.53460885382836, -108.99232380461915, -109.68685218698411, -108.36501609555184, -109.60118885953402])
        mDRQL = np.array([-88.13675905076124, -85.26639565581254, -83.86152834283286, -84.50176989389952, -83.3110157395251, -82.8940606206749, -83.17368252018316, -83.98311316712122, -85.81724257990676, -90.3356267393368])
        plt.plot(m_array, m_worst_case, "o", color = "orange", label = "Non-drql")
        plt.plot(m_array, -mDRQL, "o", color = "blue", label = "drql")

        plt.xlim(-1,10)
        plt.xticks(np.arange(0, 10, 1))
        plt.ylim(45,70)
        plt.xlabel("Values of m")
        plt.ylabel("Cost")
        plt.title("Plot of non-drql and drql for b = " + str(b_val))
        plt.legend(loc="best")
        plt.show()
        print("V = ", v)
        print("KL dist: ", mdp.get_KL_distance_from(p0_sa))

        break



# Policy evaluation sutton, få implementeret egen - skab orange