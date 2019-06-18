import os,sys;sys.path.append(os.path.abspath("/home/jpr6/pycrysfml/hklgen/"))
from os import path
import os
import gym
from gym import spaces
from gym.utils import seeding
from copy import copy
import numpy as np
import random as rand
import pickle
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axes

import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S

import  bumps.names  as bumps
import bumps.fitters as fitter
import bumps.lsqerror as lsqerr
from bumps.formatnum import format_uncertainty_pm

#from tensorforce.environments import Environment

class HklEnv(gym.Env):

    def __init__(self):


        DATAPATH = os.path.abspath("/home/jpr6/pycrysfml/hklgen/examples/sxtal")
        observedFile = os.path.join(DATAPATH,r"prnio.int")
        infoFile = os.path.join(DATAPATH,r"prnio.cfl")

        #Read data
        self.spaceGroup, self.crystalCell, self.atomList = H.readInfo(infoFile)

        #Return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
        wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=self.crystalCell)
        self.wavelength = wavelength
        self.refList = refList
        self.sfs2 = sfs2
        self.error = error
        self.tt = [H.twoTheta(H.calcS(self.crystalCell, ref.hkl), wavelength) for ref in refList]
        self.backg = None
        self.exclusions = []

        self.observation_space = spaces.MultiBinary(len(self.refList))
        self.action_space = spaces.Discrete(len(self.refList))

        self.episodeNum = 0
        self.reset()

    def epStep(self):
        self.episodeNum += 1

    def step(self, actions):

        chisq = None

        self.steps += 1

        #No repeats
        #print('actions', type(actions), actions)
        self.visited.append(self.refList[int(actions)])
        self.remainingActions.remove(actions)

        #Find the data for this hkl value and add it to the model
        self.model.refList = H.ReflectionList(self.visited)
        self.model._set_reflections()

        self.model.error.append(self.error[actions])
        self.model.tt = np.append(self.model.tt, [self.tt[actions]])

        self.observed.append(self.sfs2[actions])
        self.model._set_observations(self.observed)
        self.model.update()

        reward = -0.1

        #Need more data than parameters, have to wait to the second step to fit
        if len(self.visited) > 1:

            x, dx, chisq = self.fit(self.model)

            if (self.prevChisq != None and chisq < self.prevChisq):
                reward = 1/(chisq*10)

            self.prevChisq = chisq

        self.totReward += reward

        if (self.prevChisq != None and len(self.visited) > 50 and chisq < 5):
            return self.state, 1, True, {"chi": self.prevChisq, "z": self.model.atomListModel.atomModels[0].z.value, "hkl": self.refList[actions.tolist()].hkl}
        if (len(self.remainingActions) == 0 or self.steps > 300):
            terminal = True
        else:
            terminal = False
        return self.state, reward, terminal, {"chi": self.prevChisq, "z": self.model.atomListModel.atomModels[0].z.value, "hkl": self.refList[actions.tolist()].hkl} #, chisq, self.model.atomListModel.atomModels[0].z.value, self.refList[actions]

    def reset(self):

        #Make a cell
        cell = Mod.makeCell(self.crystalCell, self.spaceGroup.xtalSystem)

        #Define a model
        self.model = S.Model([], [], self.backg, self.wavelength, self.spaceGroup, cell,
                    self.atomList, self.exclusions,
                    scale=0.06298, error=[],  extinction=[0.0001054])

        #Set a range on the x value of the first atom in the model
        self.model.atomListModel.atomModels[0].z.value = 0.25
        self.model.atomListModel.atomModels[0].z.range(0,0.5)

        self.visited = []
        self.observed = []
        self.remainingActions = []
        for i in range(len(self.refList)):
            self.remainingActions.append(i)

        self.totReward = 0
        self.prevChisq = None
        self.steps = 0

        self.state = np.zeros(len(self.refList))
        self.stateList = []

        return self.state

    def fit(self, model):

        #Create a problem from the model with bumps,
        #then fit and solve it
        problem = bumps.FitProblem(model)
        fitted = fitter.LevenbergMarquardtFit(problem)
        x, dx = fitted.solve()
        return x, dx, problem.chisq()

    @property
    def states(self):
        return dict(shape=self.state.shape, type='float')

    @property
    def actions(self):
        return dict(num_actions=len(self.refList), type='int')
