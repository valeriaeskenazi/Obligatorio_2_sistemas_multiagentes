from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
import random
from typing import Callable

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection

def ucb(node, C=sqrt(2)) -> float:
    if node.visits == 0:
        return float("inf")  # forzar a explorar este nodo
    agent_idx = node.game.agent_name_mapping[node.agent]
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits)


def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, rollouts_steps: int=100, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        self.rollouts_steps=rollouts_steps
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> (ActionType, float):

        root = MCTSNode(parent=None, game=self.game, action=None)

        for i in range(self.simulations):

            node = root
            node.game = self.game.clone()

            #print(i)
            #node.game.render()

            # selection
            #print('selection')
            node = self.select_node(node=node)

            # expansion
            #print('expansion')
            self.expand_node(node)

            # rollout
            #print('rollout')
            rewards = self.rollout(node)

            #update values / Backprop
            #print('backprop')
            self.backprop(node, rewards)

        #print('root childs')
        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        # TODO
        # cumulate rewards and visits from node to root navigating backwards through parent
        
        while node:
            node.visits += 1
            node.cum_rewards += rewards
            node = node.parent



    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))
        for _ in range(self.rollouts):
            game = node.game.clone()
            steps = 0
            while not game.terminated() and steps < self.rollouts_steps:
                action = random.choice(game.available_actions())
                game.step(action)
                steps += 1
            rollout_rewards = np.zeros(len(game.agents))    
            if game.terminated():
                for i, agent in enumerate(game.agents):
                    rollout_rewards[i] = game.rewards[agent]
            else:
                for i, agent in enumerate(game.agents):
                    rollout_rewards[i] = game.eval(agent)
            rewards += rollout_rewards
        return rewards / self.rollouts


    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while curr_node.children:
            if curr_node.explored_children < len(curr_node.children):
                # TODO
                # set curr_node to an unvisited child
                
                child = curr_node.children[curr_node.explored_children]
                curr_node.explored_children += 1
                curr_node=child
            else:
                # TODO
                # set curr_node to a child using the selection function
                curr_node = self.selection(curr_node, self.agent)    
        return curr_node

    def expand_node(self, node) -> None:
      if not node.game.terminated():
          actions = node.game.available_actions()
          for action in actions:            
            child_game = node.game.clone()
            child_game.step(action)
            child = MCTSNode(parent=node, game=child_game, action=action)
            node.children.append(child)
          np.random.shuffle(node.children)  
   

    def action_selection(self, node: MCTSNode) -> (ActionType, float):
        action: ActionType = None
        value: float = 0
        # TODO
        # hint: return action of child with max value 
        # other alternatives could be considered
        best_child = max(node.children, key=lambda n: n.cum_rewards[node.game.agent_name_mapping[node.agent]] / n.visits)
        return best_child.action, best_child.cum_rewards[node.game.agent_name_mapping[node.agent]] / best_child.visits
   