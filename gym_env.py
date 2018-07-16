import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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


class hkl(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

     def __init__(self,  observedFile, infoFile, backgFile=None, sxtal=True):

        if sxtal:

            #Read data
            self.spaceGroup, self.crystalCell, self.atomList = H.readInfo(infoFile)

            # return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
            wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=self.crystalCell)
            self.wavelength = wavelength
            self.refList = refList
            self.sfs2 = sfs2
            self.error = error
            self.tt = [H.twoTheta(H.calcS(self.crystalCell, ref.hkl), wavelength) for ref in refList]
            self.backg = None
            self.exclusions = []

        self.reset()

    def seed(self, seed=None):
        


            

        
