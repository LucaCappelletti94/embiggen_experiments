import compress_json
from typing import Dict
from .embedding_models import AVAILABLE_MODELS
from embiggen import EdgeTransformer
from tqdm.auto import tqdm, trange
from .evaluate import evaluate
import numpy as np


def convert_to_type(value, parameter_data: Dict):
    if parameter_data["type"] == "int":
        return int(value)
    if parameter_data["type"] == "float":
        return float(value)
    if parameter_data["type"] == "str":
        return str(value)
    raise ValueError("Parameter type is not supported.")


def get_default_value(parameter_data: Dict):
    """Return the default value for given parameter."""
    return convert_to_type(parameter_data["default"], parameter_data)


def sample_value(parameter_data: Dict, number: int):
    """Return the sampled value for given parameter."""
    delta = (parameter_data["max"] - parameter_data["min"])
    extracted_value = delta * \
        (number / parameter_data["number"]) + parameter_data["min"]

    if parameter_data["scale"] == "log":
        result = np.log(extracted_value)
    elif parameter_data["scale"] == "exp":
        result = np.exp(extracted_value)
    else:
        result = extracted_value

    return convert_to_type(result, parameter_data)


def evaluate_all(
    embedding_model: str,
    results_folder: str,
    parameters_path: str,
    graph_path: str,
    graph_name: str,
    has_weights: bool,
    train_size: float = 0.8,
    random_state: int = 42,
    mlp_epochs: int = 100,
    embedder_epochs: int = 100,
    verbose: bool = False
):
    if embedding_model not in AVAILABLE_MODELS:
        raise ValueError(
            "The requested embedding model {} is not supported.".format(
                embedding_model
            )
        )
    parameters_data = compress_json.load(parameters_path)
    defaults = {
        parameter: get_default_value(parameter_data)
        for parameter, parameter_data in parameters_data.items()
    }
    for parameter, parameter_data in tqdm(parameters_data.items(), total=len(parameters_data), desc="Parameters", leave=False):
        if parameter in ("window_size", "negative_samples") and embedding_model == "GloVe":
            continue
        for i in trange(parameter_data["number"], desc="Grid search for {}".format(parameter), leave=False):
            for edge_embedding_method in tqdm(EdgeTransformer.methods.keys(), desc="Edge embedding methods", leave=False):
                evaluate(
                    results_folder=results_folder,
                    graph_path=graph_path,
                    graph_name=graph_name,
                    has_weights=has_weights,
                    embedding_model=embedding_model,
                    edge_embedding_method=edge_embedding_method,
                    **{
                        **defaults,
                        parameter: sample_value(parameter_data, i)
                    },
                    mlp_epochs=mlp_epochs,
                    embedder_epochs=embedder_epochs,
                    train_size=train_size,
                    random_state=random_state
                )
