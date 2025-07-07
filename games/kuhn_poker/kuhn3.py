import numpy as np
from numpy import ndarray
from numpy import random
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from base.game import AlternatingGame, AgentID, ActionType

class KuhnPoker3(AlternatingGame):

    def __init__(self, initial_player=None, seed=None, render_mode='human'):
        self.render_mode = render_mode

        self.seed = seed
        random.seed(seed)

        # agents
        self.agents = ["agent_" + str(r) for r in range(3)]
        self.players = [0, 1, 2]

        self.initial_player = initial_player

        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.agent_selection = None

        # actions
        self._moves = ['p', 'b']
        self._num_actions = 2
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # states
        self._max_moves = 5
        self._start = ''
        self._terminalset = set(['ppp', 'bpp', 'bbp', 'bpb', 'bbb', 
                                 'pbpp', 'pbpb', 'pbbp', 'pbbb',
                                 'ppbpp', 'ppbbp', 'ppbpb', 'ppbbb'])
        self._hist_space = Text(min_length=0, max_length=self._max_moves, charset=frozenset(self._moves))
        self._hist = None
        self._card_names = ['J', 'Q', 'K', 'A']
        self._num_cards = len(self._card_names)
        self._cards = list(range(self._num_cards))
        self._card_space = Discrete(self._num_cards)
        self._hand = None

        # observations
        self.observation_spaces = {
            agent: Dict({ 'card': self._card_space, 'hist': self._hist_space}) for agent in self.agents
        }
    
    def step(self, action: ActionType) -> None:
        agent = self.agent_selection
        # check for termination
        if (self.terminations[agent] or self.truncations[agent]):
            try:
                self._was_dead_step(action)
            except ValueError:
                print('Game has already finished - Call reset if you want to play again')
                return

        # perform step
        self._hist += self._moves[action]
        self._player = (self._player + 1) % self.num_agents
        self.agent_selection = self.agents[self._player]

        if self._hist in self._terminalset:
            self._compute_rewards()
            self.terminations = dict(map(lambda p: (p, True), self.agents))

    def _set_initial(self):
        # set initial history
        self._hist = self._start

        # deal a card to each player
        np.random.seed(self.seed)
        self._hand = np.random.choice(self._cards, size=self.num_agents, replace=False)      

        # reset agent selection
        if self.initial_player is None:
            # select random player
            self.initial_player = random.choice(self.players)
 
        self._player = self.initial_player
        self.agent_selection = self.agents[self._player]
    
    def _compute_rewards(self) -> None:
        N = self.num_agents
        _rewards = np.full(N, -1)
        bets = []
        for i, a in enumerate(self._hist):
            if a == 'b':
                bets.append(i % N)
                _rewards[i % N] -= 1
        if len(bets) == 0:
            winner = np.argmax(list(map(lambda i: self._hand[i], range(N))))
            _rewards[winner] = N-1
        else:
            bets.sort()
            best_card = np.max(list(map(lambda i: self._hand[i], bets)))
            winner = np.max(list(map(lambda i: i if self._hand[i] == best_card else 0, bets)))
            _rewards[winner] += 2 * len(bets) + (N - len(bets))
        
        self.rewards = dict(map(lambda p: (p, _rewards[self.agent_name_mapping[p]]), self.agents))

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._set_initial()

        self.observations = dict(map(lambda agent: (agent, {'card': self._hand[self.agent_name_mapping[agent]], 'hist': self._hist}), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self) -> ndarray | str | list | None:
        for agent in self.agents:
            print(agent, self._card_names[self._hand[self.agent_name_mapping[agent]]], self._hist)

    def observe(self, agent: AgentID) -> str:
        observation = str(self._hand[self.agent_name_mapping[agent]]) + self._hist
        return observation
    
    def available_actions(self):
        return list(range(self._num_actions))

    def action_move(self, action: ActionType) -> str:
        if action not in range(self._num_actions):
            raise ValueError(f"{action} is not a legal action.")
        
        return self._moves[action]