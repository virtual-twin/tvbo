#  bnm.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
r"""
Representation of a whole Brain Network Model
==============================================

$$
\dot{\psi}(x,t) = N(\psi(x,t)) + \int_{\text{local}} dx' \, g(x-x') \, S(\psi(x',t)) + \int_{\text{global}} dx' \, G(x,x') \, S(\psi(x',t- \frac{|x-x'|}{v})) + I_{\text{ext}}(x,t)
$$

"""

import numpy as np
import owlready2 as owl
import tvb.simulator.models.base
from IPython.display import Latex
from tvb.simulator.lab import *

from tvbo.export import templater
from tvbo.parse import ontology_loader

# TODO: these equations (except eq_colored) not used anywhere
eq = Latex(
    r"$\dot{\psi}(x,t)=N(\psi(x,t)) + \int\limits_{local}dx'g(x-x')S(\psi(x',t))+\int\limits_{global}dx'G(x,x')S(\psi(x',t-\frac{|x-x'|}{v}))+I_{ext}(x,t)$"
)

eq_colored = Latex(
    r"""$\dot{\psi}(x,t) = \textcolor{green}{N(\psi(x,t))} + \textcolor{red}{\int\limits_{local}dx'g(x-x')S(\psi(x',t))} +\textcolor{orange}{\int\limits_{global}dx'G(x,x')S(\psi(x',t-\frac{|x-x'|}{v}))} + \textcolor{cyan}{I_{ext}(x,t)}$"""
)

eq_box = Latex(
    r"""\begin{equation}
\dot{\psi}(x,t) = \fcolorbox{red}{white}{N($\psi$(x,t))} + \fcolorbox{orange}{white}{$\int\limits_{local}dx'g(x-x')S(\psi(x',t))$} + \fcolorbox{cyan}{white}{$\int\limits_{global}dx'G(x,x')S(\psi(x',t-\frac{|x-x'|}{v}))$} + \fcolorbox{green}{white}{$I_{ext}$(x,t)}
\end{equation}"""
)

eq_filled_box = Latex(
    r"""$\dot{\psi}(x,t) = \fcolorbox{red}{red}{N($\psi$(x,t))} + \fcolorbox{green}{green}{$\int\limits_{local}dx'g(x-x')S(\psi(x',t))$} + \fcolorbox{blue}{blue}{$\int\limits_{global}dx'G(x,x')S(\psi(x',t-\frac{|x-x'|}{v}))$} + \fcolorbox{purple}{purple}{$I_{ext}$(x,t)}$"""
)


class BrainNetworkModel:
    """
    Class that constructs a Brain Network Model, containing all elements necessary to configure a simulator:
    model, connectivity, integrator, couping and monitors.
    All these elements can be extracted from the tvbo ontology or can be provided as TVB objects
    """
    dt = 1
    sim_length = 10000

    def __init__(self, model, conn=None, integrator=None, coupling_fct=None, monitors_list=None):
        self.simulator = None

        # init model
        if model is None:
            raise ValueError("Model is required")
        elif isinstance(model, owl.ThingClass):
            model_name = model.name
            tvb_model = ontology_loader.load_tvb_model(model_name)
            tvb_model.variables_of_interest = [v.label.first().replace('_' + model.acronym.first(), '')
                                               for v in model.has_default_voi]
            self.model = tvb_model
        elif isinstance(model, tvb.simulator.models.base.Model):
            self.model = model
        elif isinstance(model, str):
            self.model = ontology_loader.load_tvb_model(model)
        else:
            raise TypeError(f'Type {type(model)} for Model is not supported')

        # init connectivity
        if conn is None:
            self.conn = connectivity.Connectivity.from_file()
        elif isinstance(conn, str):
            self.conn = connectivity.Connectivity.from_file(conn)
        elif isinstance(conn, connectivity.Connectivity):
            self.conn = conn
        else:
            raise TypeError(f'"conn" must be either a string or a TVB Connectivity object')
        self.conn.configure()

        # init integrator
        if integrator is None:
            self.integration = integrators.HeunDeterministic(dt=self.dt)   # default integrator
        elif isinstance(integrator, integrators.integration):
            integrator.dt = self.dt
            self.integration = integrator
        elif isinstance(integrator, str):
            # TODO: use ontology_loader.load_tvb_integrator after refactoring it
            integrator_obj = templater.integration2class(integrator)
            integrator_obj.dt = self.dt
            self.integration = integrator_obj
        else:
            raise TypeError(f'Type {type(integrator)} for Integrator is not supported')

        # init coupling
        if coupling_fct is None:
            self.coupling = coupling.Linear()
        elif isinstance(coupling_fct, coupling.Coupling):
            self.coupling = coupling_fct
        elif isinstance(coupling_fct, str):
            coupling_obj = ontology_loader.load_tvb_coupling_function(coupling_fct)
            self.coupling = coupling_obj
        else:
            raise TypeError(f'Type {type(coupling_fct)} for Coupling Function is not supported')

        # init monitors list
        # TODO: can we get monitors from the ontology?
        if monitors_list is None:
            default_monitor = monitors.Raw()
            default_monitor.variables_of_interest = np.array([0])
            self.monitors = [default_monitor]
        elif isinstance(monitors_list, list):
            for m in monitors_list:
                if not isinstance(m, monitors.Monitor):
                    raise TypeError(f'Type {type(m)} for Monitor is not supported')
            self.monitors = monitors_list

        # init simulator
        self.init_simulator_for_bnm()

    def init_simulator_for_bnm(self):
        sim = simulator.Simulator(
            connectivity=self.conn,
            model=self.model,
            integrator=self.integration,
            coupling=self.coupling,
            monitors=self.monitors,
            simulation_length=self.sim_length,
        )
        self.simulator = sim
