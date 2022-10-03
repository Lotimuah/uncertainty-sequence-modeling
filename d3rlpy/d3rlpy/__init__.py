import random
import os
import sys
import numpy as np
import torch

rl_baselines3_zoo_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

sys.path.append(rl_baselines3_zoo_path)

from . import (
    algos,
    constants,
    dataset,
    datasets,
    dynamics,
    envs,
    metrics,
    models,
    online,
    ope,
    preprocessing,
    wrappers,
)
from ._version import __version__


def seed(n: int) -> None:
    """Sets random seed value.

    Args:
        n (int): seed value.

    """
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic = True
