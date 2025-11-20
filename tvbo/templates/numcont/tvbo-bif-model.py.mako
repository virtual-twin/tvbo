from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
from numba import float64, guvectorize
from numcont import ContinuationPlot as cp
from numcont import ContinuationSystem as cs
from numpy import *
from tvb.basic.neotraits.api import Final, List, NArray, Range
from tvb.simulator.models.base import Model

class ${model_name}BifModel(cs.ContSystem):
    def __init__(self, N=1):
        super().__init__()

        self.AutoFortranFile = realpath("AutoFiles/model")
        self.AutoDataPath = realpath("AutoFiles/BifurcationData")

        self.SetParameterNames${tuple(parameters.keys())}
        self.SetVariableNames${tuple(state_variables.keys())}

        # Parameters
        self.t0 = 0
        self.t1 = 200

% for par, v in parameters.items():
        self.${par} = ${v['default']}
% endfor

        self.N = 1
        self.x0 = None
        self.SetN(N)

    def SetN(self, N):
        self.N = N
        self.x0 = np.zeros((6, self.N))

        ########## Basic Dfun ##########
    def dfun(self, t, x, coupling=0, local_coupling=0):

% if len(import_statements) > 0:
    % for import_statement in import_statements:
        ${import_statement}
    % endfor
% endif

        dx_dt = self.dx_dt

% for svar in state_variables:
        ${svar} = x[${list(state_variables.keys()).index(svar)}]
% endfor


## Assign parameters
% for par, v in parameters.items():
        ${par} = self.${par}
% endfor

        pi = np.pi
        exp = np.exp

## Coupling Terms
% for cterm in coupling_terms:
        ${cterm} = 0
% endfor

## Assign functions
% for var, term in ninvar_dfuns.items():
    % if var not in state_variables:
        ${var} = ${term}
    % endif
% endfor

## compute derivatives
% for svar in state_variables:
        dx_dt[${list(state_variables.keys()).index(svar)}] = ${state_variable_dfuns[svar]}
% endfor

        return dx_dt

    def Continuation(
        self,
        ICP,
        voi="X[0]",
        param_config=None,
        RL0=-0.1,
        RL1=0.5,
        kwargsEq=dict(NMX=5000, DS=0.01, NPR=50, IADS=1),
        kwargsHopf=dict(NMX=1200, DS=0.02, NPR=5, IADS=1, IAD=1, NTST=20),
        continue_hopf=True,
        run_autoIVP=True,
        t1=1000,
        tol1=1e-12,
        ):
        """
        Perform continuation analysis on the model for a specified parameter.

        This method conducts a continuation analysis on a dynamical system model
        with respect to a given parameter, using AUTO software. The process involves
        adjusting model parameters and tracking system behavior through bifurcations
        and other critical points.

        :param ICP: The index of the parameter to continue with respect to.
        :param voi: Variable of interest, default is "X[0]".
        :param param_config: Dictionary for parameter configuration, optional.
        :param RL0: Starting value for the continuation range.
        :param RL1: Ending value for the continuation range.
        :param kwargsEq: Dictionary of keyword arguments for equilibrium continuation settings.
        :param kwargsHopf: Dictionary of keyword arguments for Hopf continuation settings.
        :param continue_hopf: Boolean flag to continue Hopf bifurcation analysis.
        :param run_autoIVP: Boolean flag to run initial value problem (IVP) integration.
        :param t1: Total time for integration when running IVP.
        :param tol1: Tolerance for numerical integration.

        The method sets up the parameter configuration, clears any existing output files,
        and performs continuation using the specified settings.
        """
        if isinstance(param_config, dict):
            self.SetParamConfig(param_config)

        self.mu = RL0
        self.t1 = t1
        self.tol1 = tol1

        self.ClearFiles()
        self.Continuation1P(
            [ICP],
            f"cont_{ICP}",
            kwargsEq=kwargsEq,
            kwargsHopf=kwargsHopf,
            RL0=RL0,
            RL1=RL1,
            continue_hopf=continue_hopf,
            run_autoIVP=run_autoIVP,
        )
