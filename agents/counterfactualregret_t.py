import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, game: AlternatingGame, obs: ObsType) -> None:
        self.game = game
        self.agent = game.agent_selection
        self.obs = obs
        self.num_actions = self.game.num_actions(self.agent)
        self.cum_regrets = np.zeros(self.num_actions)
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

    def regret_matching(self):
        # Solo se consideran regrets positivos
        positive_regrets = np.maximum(self.cum_regrets, 0)
        total = np.sum(positive_regrets)
        if total > 0:
            self.curr_policy = positive_regrets / total
        else:
            # Si todos los regrets son negativos o cero, usar política uniforme
            self.curr_policy = np.full(self.num_actions, 1/self.num_actions)

    
    def update(self, utility, node_utility, probability) -> None:
        # p = índice del agente actual
        p = self.game.agent_name_mapping[self.agent]

        # Probabilidad de llegada de los demás agentes
        reach_prob_others = np.prod(probability[np.arange(len(probability)) != p])

        # Regret contrafactual: r_I += Π_{q≠p} P_q * (u - v)
        self.cum_regrets += reach_prob_others * (utility - node_utility)

        # Acumulación de política promedio: s_I += P_p * π(I)
        self.sum_policy += probability[p] * self.curr_policy

        # learn policy ponderada por la suma de políticas
        self.learned_policy = self.sum_policy/np.sum(self.sum_policy)

        # Regret matching para actualizar la política actual
        self.regret_matching()  

    def policy(self):
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            #raise ValueError('Train agent before calling action()')
            print('Node does not exist. Playing random.')
            return np.random.choice(self.game.available_actions())
    
    def train(self, niter=1000):
        for _ in range(niter):
            _ = self.cfr()

    def cfr(self):
        game = self.game.clone()
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability)

        return utility 

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        if game.done():
            return game.rewards[agent]

        obs = game.observe(game.agent_selection)

        if obs not in self.node_dict:
            self.node_dict[obs] = Node(game, obs)
        node = self.node_dict[obs]

        current_agent = game.agent_selection
        policy = node.curr_policy
        action_utils = np.zeros(node.num_actions)
        assert node.agent == current_agent, "Node agent does not match current agent in game."

        node_utility = 0.0

        for a in range(node.num_actions):
            next_game = game.clone()
            next_game.step(a)

            next_prob = probability.copy()
            if current_agent == agent:
                # No se modifica la probabilidad propia
                pass
            else:
                # Multiplico por la política actual (probabilidad de llegar al nodo desde los otros agentes)
                idx = game.agent_name_mapping[current_agent]
                next_prob[idx] *= policy[a]  

            # Llamada recursiva
            action_utils[a] = self.cfr_rec(next_game, agent, next_prob)
            node_utility += policy[a] * action_utils[a]

        if current_agent == agent:
            node.update(action_utils, node_utility, probability)

        return node_utility



    
    
