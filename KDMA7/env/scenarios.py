from typing import Callable, Sequence, Tuple
from .agents.base_agent import BaseAgent

import numpy


class rosScenario():
    def __init__(self,
        n_agents: int or Tuple[int, int],
        agent_wrapper: Callable[[], BaseAgent],
        info: list
    ):
        self.info = info
        self.n_agents = n_agents
        self.agent_wrapper = agent_wrapper
        super().__init__()
    
    def collide(self, agent0: BaseAgent, agent1: BaseAgent):
        if agent0.visible and agent1.visible:
            dist2 = (agent0.position.x-agent1.position.x)**2 + (agent0.position.y-agent1.position.y)**2
            return dist2 <= (agent0.radius+agent1.radius)**2
        return False

    def __iter__(self):
        self.counter = 0
        self.agents = []
        return self
    
    def __next__(self):
        agent = self.spawn()
        self.agents.append(agent)
        self.counter += 1
        return agent
    
    def spawn(self):
        if self.counter >= self.n_agents:
            raise StopIteration

        agent = self.agent_wrapper()

        if self.counter == 0:
            agent.goal = self.info[self.counter][5], self.info[self.counter][6]
    
        agent.position = self.info[self.counter][0], self.info[self.counter][1]
        agent.velocity = self.info[self.counter][2], self.info[self.counter][3]
        agent._orientation = self.info[self.counter][4]

        return agent
    
