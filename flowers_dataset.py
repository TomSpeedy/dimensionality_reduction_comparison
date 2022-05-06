from cProfile import label
import os
import sys
from typing import Dict, List, Optional, Sequence, TextIO
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
DATASET_DIR = "flowers"

class FLOWERS:
    H: int = 224
    W: int = 224
    C: int = 3
    LABELS: List[str] = [
        # flowers
        "daisy","dandelion", "rose", "sunflower", "tulip"      
    ]
    def __init__(self) -> None:

        self.train = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=0.2,
            subset="training",
            seed=4269,
            image_size=(FLOWERS.H, FLOWERS.W),
            batch_size = None)
        self.dev = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=0.2,
            subset="validation",
            seed = 4269,
            image_size=(FLOWERS.H, FLOWERS.W),
            batch_size = None
        )
        self.test = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            labels = None,
            validation_split=0.2,
            subset="validation",
            seed = 4269,
            image_size = (FLOWERS.H, FLOWERS.W),
            batch_size = None
        )
        
