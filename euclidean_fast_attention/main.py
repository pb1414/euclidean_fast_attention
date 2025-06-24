# main.py
from absl import app
from absl import flags
from ml_collections import config_flags
from pathlib import Path
import wandb

from euclidean_fast_attention.train import run_training


# Flag for the base configuration file
_CONFIG = config_flags.DEFINE_config_file(
    'config',
    'configs/config.py', # Must be specified
    'Path to the dataset configuration file (e.g., configs/datasets/my_dataset_config.py).'
)

# Flag for the dataset configuration file
_TRAINER_CONFIG = config_flags.DEFINE_config_file(
    'trainer_config',
    None, # Must be specified
    'Path to the dataset configuration file (e.g., configs/datasets/my_dataset_config.py).'
)

# Flag for the model configuration file
_MODEL_CONFIG = config_flags.DEFINE_config_file(
    'model_config',
    None, # Must be specified
    'Path to the model configuration file (e.g., configs/models/model_a_config.py).'
)

# Flag for the optimizer configuration file
_OPTIMIZER_CONFIG = config_flags.DEFINE_config_file(
    'optimizer_config',
    'configs/optimizer/default.py', # Must be specified
    'Path to the model configuration file (e.g., configs/models/model_a_config.py).'
)

# You can add other general flags like run name
FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', 'default_run', 'Workdir of the current run.')


def main(_):
    # Load base configuration
    config = _CONFIG.value
    config.unlock()
    config.workdir = FLAGS.workdir
    
    # Load dataset configuration
    train_config = _TRAINER_CONFIG.value

    # Load model configuration
    model_config = _MODEL_CONFIG.value

    # Load optimizer configuration
    opt_config = _OPTIMIZER_CONFIG.value

    config.trainer = train_config
    config.model = model_config
    config.optimizer = opt_config
    
    print(f"\n--- Combined Configuration ---")
    print(config.to_json(indent=2))

    assert config.workdir is not None

    if Path(config.workdir).resolve().exists():
        raise RuntimeError(
            f'Specified workdir {Path(config.workdir).resolve()} already exists.'
        )

    wandb.init(
        config=config.to_dict(),
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
    )

    run_training(
        config=config,
        workdir=config.workdir
    )

if __name__ == '__main__':
    app.run(main)
