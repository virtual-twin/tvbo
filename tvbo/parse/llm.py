#
# Module: llm.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
import json
import os
from collections import OrderedDict

import yaml
from openai import OpenAI

from tvbo.datamodel.tvbopydantic import (
    SimulationStudy,
    SimulationExperiment,
)  # TODO: import from tvbo-datamodel later

default_system_content = """
You are an expert assistant tasked with extracting data from scientific text and converting it into JSON format that is fully compliant with the SimulationStudy model schema. Your goal is to ensure accuracy, completeness, and adherence to the specified schema.

1. **Experiments**:
   - Thoroughly scan the text for all sections, captions, or contexts that describe or hint at an experiment.
   - Extract each experiment and assign it a clear, one-word name derived from the preceding caption, heading, or the experiment's core concept.
   - Ensure no experiment is overlooked, even if it is not explicitly labeled as such. Consider any mention of a study, simulation, or test as a potential experiment.
   - Document each experiment separately, ensuring that all associated parameters, connectivity, coupling, and integration details are accurately captured.

2. **Models**: If the text references any model from the following list, set it as the 'label' in the 'derived_from_model' attribute. Ensure that this attribute complies with the NeuralMassModel schema, including:
   - 'label': The model name (e.g., 'Kuramoto')
   - Other required fields according to the schema (e.g., 'iri', 'modified', 'output_transform').
   The models to check include: ['GastSchmidtKnosche_SD', 'CoombesByrne', 'CoombesByrne2D', 'DumontGutkin', 'Epileptor2D', 'Epileptor5D', 'EpileptorRestingState', 'GastSchmidtKnosche_SF', 'Generic2dOscillator', 'GenericLinear', 'Hopfield', 'JansenRit', 'KIonEx', 'Kuramoto', 'LarterBreakspear', 'MontbrioPazoRoxin', 'ReducedWongWang', 'ReducedWongWangExcInh', 'SupHopf', 'WilsonCowan', 'ZerlautAdaptationFirstOrder', 'ZetterbergJansen'].

3. **Parameters**: Extract all variables used in equations as parameters, ensuring:
   - 'label': The variable name.
   - 'value': Only set if it is a single real number (**NO ARRAYS OR LISTS**). It must have the dtype float.
   - 'domain': Use attributes lo, hi, step for ranges if applicable:
    ```yaml
    domain:
        - lo: 0
        - hi: 1
        - step: 0.1
    ```
   - 'unit': Include if relevant.
   - 'definition': Provide a clear and precise description.
   - **Arrays or matrices**: Do not assign to 'value'. Instead, describe them in the 'definition' and provide relevant details (e.g., dimensions, contents).

4. **Equations**: Extract equations formatted for compatibility with Python's `sympy` library. Include:
   - 'label': A descriptive name for the equation.
   - 'lefthandside': The variable on the left-hand side.
   - 'righthandside': The right-hand side of the equation.
   - 'definition': A clear explanation of the equation's purpose.
   Ensure all variables in the equations have corresponding parameters.

5. **Connectivity and Coupling**: For each experiment, extract the connectivity and coupling information directly relevant to the study:
   - **Connectivity**: Include details about parcellation (label, data source, number of regions) and weights. THIS IS ESSENTIAL!
   - **Coupling**: Extract coupling parameters separately, including information on free parameters, their domains, and any reported optima.

6. **Integration**: Document the integration method used in the study, including:
   - Noise settings.
   - Integration time and any specific parameters.

Avoid including any fields with null, undefined, or placeholder values. If any required data is missing, simply omit it. Ensure that the resulting JSON strictly adheres to the SimulationStudy model schema.
"""

default_user_content = """
Please extract all relevant scientific data from the following text and convert it into JSON format, compliant with the SimulationStudy model schema.

1. **Experiments**: Parse and document all experiments mentioned in the text. Assign a precise, one-word name, ideally derived from the preceding caption or context. Ensure no experiments are missed.

2. **Parameters**: For each parameter, ensure the following:
   - 'label' (the variable name, typically the mathematical symbol)
   - 'value' (numerical value, **ONLY IF IT'S A SINGLE REAL NUMBER, NO LISTS!**)
   - 'domain' (use `Range(lo=, hi=, step=)` if applicable)
   - 'unit' (if applicable)
   - 'definition'
   - **For arrays or matrices**: Do not use 'value'. Instead, describe them in 'definition' and provide relevant details (e.g., dimensions, contents).

3. **Equations**: Extract each equation with:
   - 'label'
   - 'lefthandside' (variable on the left-hand side)
   - 'righthandside' (right-hand side of the equation)
   - 'definition'
   Ensure all variables used in the equations are accounted for as parameters and formatted for Python's `sympy` library.

4. **Connectivity**: Within each experiment, extract the specific connectivity information used, including:
   - Parcellation (label, data source, number of regions)
   - Weights

5. **Coupling**: Extract coupling parameters separately, ensuring:
   - Any free parameters are included
   - Their domains are specified
   - Reported optima are documented if available

6. **Model Derivation**: If a known model is mentioned in the text, include it as the value for the 'label' key under the 'derived_from_model' attribute, ensuring it complies with the NeuralMassModel schema. For example:
   ```yaml
   derived_from_model:
     label: Kuramoto
    ```
Ensure all required fields as per the schema are included.

	7.	Integration and Noise: Include details on the integration method used in the study, along with any specific parameters or settings for:
	•	Noise
	•	Integration time (‘duration’)
	8.	Validation: Verify that all variables on the righthandside of an equation are accounted for as parameters. Avoid including any null or placeholder fields. IMPORTANT: Ensure the data format is strictly compliant with the SimulationStudy model schema. DO NOT COPY DATA OR VALUES FROM ONE EXPERIMENT TO ANOTHER UNLESS THEY USE THE SAME MODEL AND PARAMETERS.

The JSON output must be valid, complete, and fully compliant with the SimulationStudy model schema, suitable for evaluation with Python. !IMPORTANT MAKE SURE ALL PARAMETER VALUES ARE FLOATS AND NO LISTS!
"""

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Define the function schema
functions = [
    {
        "name": "extract_data",
        "description": "Simulation study",
        "parameters": SimulationStudy.model_json_schema(),
    },
]


# Define messages
def define_promt(
    text, system_content="default", user_content="default", expected_experiments=None
):
    if system_content == "default":
        system_content = default_system_content
    if user_content == "default":
        user_content = default_user_content

    if expected_experiments:
        experiment_text = f"\n THERE ARE {expected_experiments} EXPECTED EXPERIMENTS. Please ensure that all experiments are extracted and documented."
    else:
        experiment_text = ""

    messages = [
        {
            "role": "system",
            "content": system_content + experiment_text,
        },
        {
            "role": "user",
            "content": (user_content + text),
        },
    ]
    return messages


def extract_data(text, model="gpt-4o", expected_experiments=None):
    """
    Extracts data using the OpenAI API.
    Args:
        text (str): The input text to be processed.
        model (str, optional): The model to be used for processing. Defaults to "gpt-4o".
    Returns:
        dict: The extracted data instance.
    """

    # Define the prompt
    prompt = define_promt(text, expected_experiments=expected_experiments)

    # Call the OpenAI API
    chat_completion = client.chat.completions.create(
        messages=prompt,
        model=model,
        functions=functions,
        function_call={"name": "extract_data"},  # Specify the function to be called
        seed=1312,  # Set seed for reproducibility
        temperature=0.0,  # Set temperature to 0 for deterministic output
        top_p=1.0,  # Set top_p to 1 to consider all tokens in the distribution
        timeout=1000,  # Set timeout to 1000 seconds
    )

    # Extract the data

    # Convert the string to a proper JSON format if necessary
    json_str = (
        chat_completion.choices[0]
        .message.function_call.arguments.replace("None", "null")
        .replace("True", "true")
        .replace("False", "false")
    )

    # Parse the JSON string into a Python dictionary
    try:
        data_instance = json.loads(json_str)
    except:
        try:
            data_instance = eval(json_str)
        except:
            raise ValueError(
                f"Could not parse the JSON string into a Python dictionary. \n\n{json_str}"
            )

    return data_instance


def extract_parameters(text, model="gpt-3.5-turbo", simulation_study_data=None):
    system_content = "You are an assistant that extracts parameters from scientific text and converts them into a structured JSON format."

    if simulation_study_data:
        simulation_study_content = f"""
    Map them to the correct models within the provided simulation study data. The output should be in JSON format, preserving the **whole** structure of the simulation study.

    Simulation Study: {json.dumps(simulation_study_data, indent=2)}
    """
    else:
        simulation_study_content = ""

    user_content = f"""
    Please extract **all** parameters from the following text and convert them into JSON format with the following fields:
    - label: The parameter name or symbol. !IMPORTANT: WRITE OUT UNICODE CHARACTERS (e.g. 𝜈 to nu)!
    - value: The numerical value of the parameter.
    - unit: The unit of the parameter.
    - definition: A brief description of the parameter.
    Ensure that **all** parameters are extracted from the text.

    {simulation_study_content}

    Text: {text}
    """
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    functions = [
        {
            "name": "extract_data",
            "description": "Model schema listing all parameters for the TVB model.",
            "parameters": SimulationExperiment.model_json_schema(),
        },
    ]

    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Call the OpenAI API
    chat_completion = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.0,
        functions=functions,
        function_call={"name": "extract_data"},
        seed=1312,
        top_p=1.0,
    )

    # Extract the JSON data from the response
    json_str = (
        chat_completion.choices[0]
        .message.function_call.arguments.replace("None", "null")
        .replace("True", "true")
        .replace("False", "false")
    )

    # Parse the JSON string into a Python dictionary
    data_instance = json.loads(json_str)
    return data_instance


# Define the order of keys
sort_order = ["id", "model", "connectivity", "coupling", "integration"]


def convert2yaml(data_instance, fname=None):
    """
    Converts the data instance to a YAML file.
    Args:
        data_instance (dict): The data instance to be converted.
    Returns:
        str: The YAML representation of the data instance.
    """
    yaml_str = yaml.dump(
        data_instance, default_flow_style=False, allow_unicode=True, sort_keys=True
    )

    if fname:
        with open(fname, "w") as f:
            f.write(yaml_str)
    else:
        return yaml_str
