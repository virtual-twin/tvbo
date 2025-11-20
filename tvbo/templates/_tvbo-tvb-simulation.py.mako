# -*- coding: utf-8 -*-

"""
Simulator for TVB
=================
This part of the script is used to define the simulator for TVB.
"""

def load_connectivity(weights_path, lengths_path):
    """
    Load the weights and lengths from the provided CSV files and create the Connectivity object.

    Args:
        weights_path (str): Path to the CSV file containing the weights matrix.
        lengths_path (str): Path to the CSV file containing the tract lengths matrix.

    Returns:
        Connectivity: Configured TVB Connectivity object.
    """
    weights = pd.read_csv(weights_path, header=None).values
    lengths = pd.read_csv(lengths_path, header=None).values

    SC = Connectivity(
        weights=weights,
        tract_lengths=lengths,
    )
    SC.set_centres(np.repeat((0, 0, 0), weights.shape[0]), weights.shape[0])
    SC.create_region_labels()
    SC.configure()

    return SC

def define_simulation(weights_path, lengths_path, simulation_length):
    """
    Define and configure the TVB simulation based on the provided connectivity and simulation parameters.

    Args:
        weights_path (str): Path to the CSV file containing the weights matrix.
        lengths_path (str): Path to the CSV file containing the tract lengths matrix.
        simulation_length (float): Duration of the simulation (in ms).

    Returns:
        Simulator: Configured TVB Simulator object ready to run the simulation.
    """
    SC = load_connectivity(weights_path, lengths_path)
    sim = Simulator(
        model=${model["name"] if model else "LocalNeuralDynamics()"},
        connectivity=SC,
        coupling=${coupling["name"] if coupling else "Coupling()"},
        integrator=${integration["method"] if integration else "Integrator()"},
        simulation_length=simulation_length,
    )
    return sim

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description="Run TVB simulator with custom weights and lengths."
    )

    parser.add_argument(
        "--fweights",
        type=str,
        ${f"default={fweights}" if fweights else "required=True"},
        help="Path to the CSV file containing the weights matrix.",
    )
    parser.add_argument(
        "--flengths",
        type=str,
        ${f"default={flengths}" if flengths else "required=True"},
        help="Path to the CSV file containing the tract lengths matrix.",
    )
    parser.add_argument(
        "--simulation_length",
        type=float,
        default=1000,
        help="Simulation length for the TVB simulator. Default is 1000.",
    )
    args = parser.parse_args()

    # Run the simulation
    sim = define_simulation(args.fweights, args.flengths, args.simulation_length)
    sim.configure()
    sim.run()
