import random
import numpy as np
from config import get_config
import inspect
import os

def set_seed():
    seed = get_config("seed")
    random.seed(seed)
    np.random.seed(seed)
    
    # Get the name of the file that called this function
    frame = inspect.stack()[1]
    module_name = os.path.basename(frame.filename)
    
    print(f"[{module_name}] Seed set to: {seed}")