import argparse

from config import get_config
from dataset import data_loader
from neural_methods import trainer


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/infer_configs/PURE_UNSUPERVISED.yaml", type=str, help="The name of the model.")
    return parser


# parse arguments.
parser = argparse.ArgumentParser()
parser = add_args(parser)
parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
args = parser.parse_args()

# configurations.
config = get_config(args)

logging_path = config.UNSUPERVISED.DATA.CACHED_PATH.split("/")[-1]

print(logging_path)
