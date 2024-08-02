""""
This is a wrapper for EpiSim.jl

It handles the file writing and reading for executing the model: config files and model state.
We do it like this to get the precompile performance benefits but it has some downsides, mainly:

- we cannot run two instances of the model at the same time
- we need to write the config json file every time we change the params
- we need to write the model state .nc file for every step !!

The alternative is to call functions from EpiSim.jl directly but this also has downsides:

- startup cost every single time we run a step
- we need to change EpiSim.jl to enable this direct access
- we need to marshall data types between python and julia, particularly the model state (big arrays!)
"""

import os, sys
import json
import subprocess
import logging
import pandas as pd
import uuid
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(logging.StreamHandler(sys.stdout))

logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

class MMCACovid19:
    def __init__(self, executable_path, config, data_folder, instance_folder, initial_conditions=None):
        # assert the compiled executable exists
        assert os.path.exists(executable_path)
        assert os.path.isfile(executable_path)
        assert os.access(executable_path, os.X_OK)

        assert os.path.exists(data_folder)
        assert os.path.exists(instance_folder)

        self.executable_path = executable_path
        self.instance_folder = instance_folder
        self.uuid = str(uuid.uuid4())
        self.model_state_folder = os.path.join(instance_folder, self.uuid)
        os.makedirs(self.model_state_folder, exist_ok=False)

        config_path = MMCACovid19.handle_config_input(self.model_state_folder, config)

        self.config_path = config_path
        self.data_folder = data_folder

        self.model_state = initial_conditions
        if initial_conditions:
            # Copy initial conditions to the unique folder
            new_initial_conditions = os.path.join(self.model_state_folder, os.path.basename(initial_conditions))
            shutil.copy(initial_conditions, new_initial_conditions)
            self.model_state = new_initial_conditions

        logger.info(f"Model wrapper init complete. UUID: {self.uuid}")
        return self.uuid

    def step(self, start_date, length_days):
        """
        Run the model for a given number of days starting from a given start date.
        Updates the config and the model state, then calls the simulator.

        Returns the model state and the next start date.
        """

        end_date = date_addition(start_date, length_days - 1)

        logger.debug(f"Running model from {start_date} to {end_date}")
        self.run_model(
            length_days=length_days,
            start_date=start_date,
            end_date=end_date,
            model_state=self.model_state,
        )

        self.model_state = self.model_state_filename(end_date)

        logger.debug(f"Step complete")
        return self.model_state, date_addition(end_date, 1)

    def update_config(self, config):
        self.config_path = self.handle_config_input(self.model_state_folder, config)

    def run_model(self, length_days, start_date, end_date, model_state=None):
        """
        Run the compiled model (eg. MMCACovid19Vac).
        Provide your config as a dict or as a path to a json file.
        """

        cmd = [self.executable_path]
        cmd.extend(["--config", self.config_path])
        cmd.extend(["--data-folder", self.data_folder])
        cmd.extend(["--instance-folder", self.model_state_folder])

        if model_state:
            cmd.extend(["--initial-conditions", model_state])

        # save the model state at the final day of the run
        cmd.extend(["--export-compartments-time-t", str(length_days)])
        # this overrides the config.json !!
        cmd.extend(["--start-date", start_date])
        cmd.extend(["--end-date", end_date])

        cmdstr = " ".join(cmd)
        logger.debug(f"Running command:\n{cmdstr}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Model execution failed: {result.stderr}")

        return result.stdout

    def model_state_filename(self, end_date):
        return os.path.join(self.model_state_folder, "output", f"compartments_t_{end_date}.h5")

    def update_model_state(self, end_date):
        self.model_state = self.model_state_filename(end_date)
        pass

    @staticmethod
    def handle_config_input(model_state_folder, config):
        """
        Note that because we use a static file name you cannot run two instances of MMCACovid19 at the same time.
        TODO: enable that with unique config file names (?)
        """
        if isinstance(config, dict):
            # write our own config file for the model to use
            config_path = os.path.join(model_state_folder, 'config_auto_py.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        elif isinstance(config, str) and os.path.exists(config):
            config_path = os.path.join(model_state_folder, os.path.basename(config))
            shutil.copy(config, config_path)
        else:
            raise ValueError(f"Invalid config: {config}")

        logger.debug(f"Using config at: {config_path}")
        return config_path

# MMCACovid19 class end


######################
# utils
######################

def date_addition(start_date, length_days):
    """Calculate the end date given a start date and number of days."""
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=length_days)
    return end.strftime('%Y-%m-%d')

def pardir(): return os.path.join(os.path.dirname(__file__), "..")


######################
# example usage
######################

def run_model_example():
    executable_path = os.path.join(pardir(), "episim")

    initial_conditions = os.path.join(pardir(), "models/mitma/initial_conditions.nc")

    # read the config file sample to dict
    with open(os.path.join(pardir(), "models/mitma/config.json"), 'r') as f:
        config = json.load(f)

    data_folder = os.path.join(pardir(), "models/mitma")
    instance_folder = os.path.join(pardir(), "runs")

    model = MMCACovid19(
        executable_path, config, data_folder, instance_folder, initial_conditions
    )

    logger.info("Running")
    output = model.run_model(length_days=1, start_date="2023-01-01", end_date="2023-01-02")
    logger.info(output)
    logger.info("Example done")

def agent_flow_example():
    "Run steps and update the policy"
    executable_path = os.path.join(pardir(), "episim")

    initial_conditions = os.path.join(pardir(), "models/mitma/initial_conditions.nc")

    # read the config file sample to dict
    with open(os.path.join(pardir(), "models/mitma/config.json"), 'r') as f:
        config = json.load(f)

    data_folder = os.path.join(pardir(), "models/mitma")
    instance_folder = os.path.join(pardir(), "runs")

    model = MMCACovid19(
        executable_path, config, data_folder, instance_folder, initial_conditions
    )
    logger.debug("debug Model wrapper init complete")

    start_date="2023-01-01"
    logger.info(f"First date: {start_date}")
    current_date = start_date
    for i in range(10):
        new_state, next_date = model.step(start_date=current_date, length_days=10)

        # update the policy
        # increase the level of lockdown by 5% at each iteration
        config["NPI"]["κ₀s"] = [ config["NPI"]["κ₀s"][0] * (1 - 0.05) ]
        model.update_config(config)

        logger.debug(f"Iteration {i+1} - Model state: {new_state}")
        logger.info(f"Iteration {i+1} - Next date: {next_date}")
        current_date = next_date

    logger.info("Example done")


if __name__ == "__main__":
    agent_flow_example()