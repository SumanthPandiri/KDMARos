from typing import Sequence
import numpy
import torch
import config
from env.agents.base_agent import BaseAgent
from .networks import Policy

class DLAgent(BaseAgent):

    def __init__(self, expert: Policy or Sequence[Policy] = None, model: Policy = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.expert = expert
        self.expert_ob_ = None
        
    def observe(self, env):
        self.expert_ob = self.expert_ob_
        
        dpx, dpy = self.goal.x-self.position.x, self.goal.y-self.position.y
        dist = (dpx*dpx+dpy*dpy)**0.5
        
        n = []
        for neighbor in env.agents:
            if not self.observable(neighbor): continue
            n.extend([
                neighbor.position.x - self.position.x,
                neighbor.position.y - self.position.y,
                neighbor.velocity.x - self.velocity.x,
                neighbor.velocity.y - self.velocity.y
            ])
        
        if self.expert:            
            self.expert_ob_ = [
                dpx, dpy,
                self.velocity.x, self.velocity.y,
                0., 0., 0., 0. # dummy neighbor representing agent itself
            ] + n
                   
        heading = numpy.arctan2(dpy, dpx)
        c = numpy.cos(-heading)
        s = numpy.sin(-heading)
        R = numpy.asarray([
            [c, -s],
            [s,  c]
        ])
        if n: n = (R @ numpy.array(n).reshape(-1, 2, 1)).reshape(-1).tolist()
        v = R @ [self.velocity.x, self.velocity.y]
        if dist > self.observe_radius: dist = self.observe_radius
        ob = [
            dist, v[0], v[1],
            0., 0., 0., 0.
        ] + n
        return ob

    def act(self, s, env):
        #print(s)
        if s is None: return None

        if len(s) == 2: # from action
            a = s
        elif len(s) > 2:  # infer from states
            a = self.model(
                self.model.placeholder(s[:self.model.agent_dim]),
                self.model.placeholder(s[self.model.agent_dim:]).view(-1, 4)
            ).cpu().tolist()
        else:
            raise ValueError

        dp = self.goal.x-self.position.x, self.goal.y-self.position.y
        heading = numpy.arctan2(dp[1], dp[0])
        c = numpy.cos(heading)
        s = numpy.sin(heading)
        R = numpy.asarray([
            [c, -s],
            [s,  c]
        ])
        
        theta = numpy.clip(a[0], -numpy.pi, numpy.pi)
        # v = numpy.max(a[1], 0)
        a = numpy.cos(theta)*a[1], numpy.sin(theta)*a[1]

        a = R @ a
    
        orient = numpy.arctan2(self.velocity.y, self.velocity.x)
        lin_vel = numpy.linalg.norm(a)
        ang_vel = numpy.arctan2(a[1], a[0]) - orient
        if ang_vel > numpy.pi:
            ang_vel -= numpy.pi*2
        elif ang_vel < -numpy.pi:
            ang_vel += numpy.pi*2

        if config.ACC == True:
            return [(lin_vel-self._lin_vel)*env.fps, (ang_vel-self._ang_vel)*env.fps]
        
        return lin_vel, ang_vel
        

        vx_, vy_ = a[0], a[1]
        return [(vx_-self.velocity[0])*env.fps, (vy_-self.velocity[1])*env.fps]

