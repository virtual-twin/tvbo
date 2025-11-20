#
# Module: lems_loader.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
This module deals with loading elements from LEMS-generated files.
"""
from lems import api as lems

from tvbo import equations
from tvbo.knowledge import ontology


def lems_model_info(model):
    model_definition = model.description
    derivatives = model.component_types["derivatives"]
    # TODO: dpms not used, remove?
    dpms = derivatives.derived_parameters
    params = {
        k: dict(
            description=v.symbol,  # v.description,  # TODO: fix hack
            symbol=k,
            value=eval(v.value) if v.value else v.numeric_value,
            range=ontology.get_range(v.dimension),
        )
        for k, v in derivatives.constants.items()
    }
    vois = derivatives.exposures
    state_variables = {
        k: dict(
            label=k,
            symbol=k,
            range=ontology.get_range(v.dimension),
            boundaries=ontology.get_range(v.exposure) if v.exposure else None,
        )
        for k, v in derivatives.dynamics.state_variables.items()
    }

    ninvars = {
        k: {
            "range": v.dimension,
            "exposure": v.exposure,
            "value": v.value,
            "select": v.select,
            "required": v.required,
            # "symbol": v.symbol,
        }
        for k, v in derivatives.dynamics.derived_variables.items()
    }

    sv_dfuns = {
        sv: f.value.replace("{", "(").replace("}", ")")
        for f in derivatives.dynamics.time_derivatives
        for sv in state_variables
        if sv in f.variable
    }

    coupling_functions = {
        k: {
            "parameters": cf.parameters,
            "constants": cf.constants,
            "derived_parameters": {
                dp_k: dp_v.value for dp_k, dp_v in cf.derived_parameters.items()
            },
            "cfuns": {
                dv_k: dv_v.value for dv_k, dv_v in cf.dynamics.derived_variables.items()
            },
        }
        for k, cf in model.component_types.items()
        if "coupling" in k.lower()
    }

    return dict(
        state_variables=state_variables,
        parameters=params,
        non_integrated_variables=ninvars,
        state_variable_dfuns=sv_dfuns,
        vois=vois,
        model_definition=model_definition,
        coupling_functions=coupling_functions,
    )


def load_lems_model(lems_file):
    model = lems.Model()
    model.import_from_file(lems_file)
    return lems_model_info(model)


def import_lems_model(lems_file, model_name):
    onto = ontology.onto
    data = load_lems_model(lems_file)

    acr = ontology.create_acronym(model_name)
    model_suffix = "_" + acr

    def create_onto_subclass(name, base_class, properties, model_class):
        with onto:
            new_class = type(
                name,
                (
                    model_class,
                    base_class,
                ),
                {},
            )
            for prop_name, prop_value in properties.items():
                if isinstance(prop_value, list):
                    getattr(new_class, prop_name).extend(prop_value)
                else:
                    getattr(new_class, prop_name).append(prop_value)
            return new_class

    # Create the main model class
    with onto:
        model_class = type(
            model_name,
            (onto.NeuralMassModel,),
            {
                "label": model_name,
                "definition": data["model_definition"],
                "acronym": acr,
            },
        )
        model_class.VOIs = ", ".join(list(data["vois"].keys()))

    # Adding state variables as subclasses of the model
    for sv_name, sv_info in data.get("state_variables", {}).items():
        properties = {
            "label": [sv_info["label"] + model_suffix],
            "symbol": [sv_info["symbol"]],
            "stateVariableRange": [", ".join(sv_info["range"])],
        }
        if sv_info["boundaries"]:
            properties["stateVariableBoundaries"] = [", ".join(sv_info["boundaries"])]
        sv_class = create_onto_subclass(
            sv_name + model_suffix, onto.StateVariable, properties, model_class
        )
        model_class.has_state_variable.append(sv_class)

    # Adding parameters as subclasses of the model
    for param_name, param_info in data.get("parameters", {}).items():
        properties = {
            "label": [param_name + model_suffix],
            "definition": [
                param_info["description"] if param_info["description"] else ""
            ],
            "symbol": [param_info["symbol"] if param_info["symbol"] else param_name],
            "defaultValue": [param_info["value"]],
            "range": [", ".join(param_info["range"])],
        }
        param_class = create_onto_subclass(
            param_name + model_suffix, onto.Parameter, properties, model_class
        )
        model_class.has_parameter.append(param_class)

    # Adding non-integrated variables as subclasses of the model
    for niv_name, niv_info in data.get("non_integrated_variables", {}).items():
        # for k, v in niv_info.items():
        #     print(k, v)
        properties = {
            "label": [niv_name + model_suffix],
            "symbol": niv_name,
            "value": niv_info["value"],
        }
        create_onto_subclass(niv_name, onto.Function, properties, model_class)

    # Adding state variable dfuns as subclasses of the model
    for dfun_name, dfun_value in data.get("state_variable_dfuns", {}).items():
        label = dfun_name + "dot" + model_suffix
        td_cls = create_onto_subclass(
            label,
            onto.TimeDerivative,
            {
                "label": [label],
                "value": [dfun_value],
                "symbol": [rf"\dot{{{dfun_name}}}"],
            },
            model_class,
        )
        td_cls.is_a.append(onto[dfun_name + model_suffix])
        td_cls.is_derivative_of.append(onto[dfun_name + model_suffix])
        onto[dfun_name + model_suffix].has_derivative.append(td_cls)

    # Coupling Functions
    for cf_name, cf in data.get("coupling_functions", {}).items():
        cf_name = cf_name + model_suffix.replace("_", "_c")
        # print(cf_name)
        with onto:
            cf_class = type(
                cf_name,
                (onto.Coupling,),
                {
                    "label": cf_name,
                    "definition": data["model_definition"],
                    "acronym": "c" + acr,
                },
            )

        for param, value in {**cf["constants"], **cf["parameters"]}.items():
            param_name = param + model_suffix.replace("_", "_c")
            properties = {
                "label": [param + model_suffix.replace("_", "_c")],
                "symbol": [value.name],
                "defaultValue": [value.value] if value.value else "None",
            }
            cparam_class = create_onto_subclass(
                param_name, onto.Parameter, properties, cf_class
            )
            cf_class.has_parameter.append(cparam_class)

        for param, value in cf["derived_parameters"].items():
            if len(ontology.onto.search(label=param)) == 1:
                with ontology.onto:
                    ontology.onto[param].is_a.append(model_class)

        pre = cf["cfuns"]["pre"]
        create_onto_subclass(
            "pre" + model_suffix.replace("_", "_c"),
            onto.Fpre,
            {"value": pre},
            cf_class,
        )

        post = cf["cfuns"]["post"]
        create_onto_subclass(
            "post" + model_suffix.replace("_", "_c"),
            onto.Fpost,
            {"value": post},
            cf_class,
        )

    equations.update_mathematical_relationships(model_class)
    return model_class, cf_class
