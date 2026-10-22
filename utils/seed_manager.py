import os
import random
import numpy as np


def seed_everything(seed: int = 42):
    """
    Lock all random seeds in the main process to ensure absolute determinism during environment initialization
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If future research incorporates deep learning frameworks, this can be seamlessly extended to include functions such as `torch.manual_seed(seed)`.
    print(f"[Seed Manager] Global seed locked to {seed} in main process.")


def worker_init_fn(worker_id: int, base_seed: int = 42):
    """
    Initialization function for multi-process workers.
    Assigns an orthogonal seed to each independent process, ensuring that the random number streams used in parallel computation are both non-overlapping and perfectly reproducible.
    Intended to be called within the `initializer` argument of `multiprocessing.Pool`.
    """
    # Generate independent seeds for child processes by combining `worker_id` and `base_seed` with a large prime number
    process_seed = (base_seed + worker_id * 10007) % (2 ** 32 - 1)

    random.seed(process_seed)
    np.random.seed(process_seed)
    os.environ['PYTHONHASHSEED'] = str(process_seed)