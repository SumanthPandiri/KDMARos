import numpy
import torch

from env.multi_agent_env import DecentralizedMultiAgentEnv


class Env(DecentralizedMultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.expert = None

    def observe(self):
        obs = []
        # remove successful agents
        for agent in self.agents:
            if agent in self.info["arrived_agents"]:
                agent.visible = False
        for agent in self.agents:
            if agent in self.info["collided_agents"] or agent in self.info["arrived_agents"]:
                obs.append(None) # stop agent
            else:
                obs.append(agent.observe(self))
        return obs

    def reset(self):
        states = super().reset()
        self.velocity = [(agent.velocity.x, agent.velocity.y) for agent in self.agents]
        self.position = [(agent.position.x, agent.position.y) for agent in self.agents]
        return states

    def step(self, action, n):
        self.action = action
        states, rews, terminal, info = super().step(self.action, n)
        for i, agent in enumerate(self.agents):
            self.velocity[i] = (agent.velocity.x, agent.velocity.y)
            self.position[i] = (agent.position.x, agent.position.y)
        return states, rews, terminal, info

    def reward(self, idx, agent):
        if agent in self.info["collided_agents"]:
            r = -0.25
            return r
        if agent in self.info["arrived_agents"]:
            r = 1
            return r
        r = 0

        vx = (agent.position.x - self.position[idx][0])*self.fps
        vy = (agent.position.y - self.position[idx][1])*self.fps

        dt = 1./self.fps
        v_pref = agent.preferred_velocity(dt, start=self.position[idx])
        agent.vpref = v_pref
        e0 = ((v_pref[0]-vx)**2+(v_pref[1]-vy)**2)**0.5
        agent.vr = 0.02*numpy.exp(-0.85*e0)
        r += agent.vr

        return r

