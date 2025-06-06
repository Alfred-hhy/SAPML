from Compiler import ml
from Compiler.types import MultiArray, sfix, sint, Array, Matrix
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

from typing import List, Optional

import torch.nn as nn
import torch

import numpy as np
import time

class AdultInputLoader(AbstractInputLoader):

    def __init__(self, dataset, n_train_samples: List[int], n_wanted_train_samples: List[int], n_wanted_trigger_samples: int, n_wanted_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool, consistency_check: Optional[str], sha3_approx_factor, input_shape_size: int, load_model_weights: bool = True):

        INPUT_FEATURES = 91
        self._dataset = dataset

        train_dataset_size = sum(n_wanted_train_samples)
        print(f"Compile loading Adult data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_wanted_trigger_samples} audit trigger samples")
        print(f"  {n_wanted_test_samples} test samples (not audit relevant)")

        self._train_samples = Matrix(train_dataset_size, INPUT_FEATURES, sfix)
        self._train_labels = sint.Tensor([train_dataset_size])

        self._audit_trigger_samples = sfix.Tensor([n_wanted_trigger_samples, INPUT_FEATURES])
        self._audit_trigger_mislabels = sint.Tensor([n_wanted_trigger_samples])

        self._test_samples = MultiArray([n_wanted_test_samples, INPUT_FEATURES], sfix)
        self._test_labels = sint.Tensor([n_wanted_test_samples])


        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_pytorch(dataset, n_train_samples, debug=debug)
        self._load_input_data_pytorch(train_datasets, backdoor_dataset, test_dataset,
                                      n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_wanted_test_samples,
                                      audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug, consistency_check=consistency_check, load_model_weights=load_model_weights,
                                      sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)


    def model_latent_space_layer(self):
        expected_latent_space_size = 32
        return self._model.layers[-3], expected_latent_space_size


    def model_layers(self):
        layers = [
            ml.keras.layers.Dense(32, activation='relu'),
            ml.keras.layers.Dense(2, activation='softmax')
        ]
        return layers

    def one_hot_labels(self):
        return False

    def _load_model(self, input_shape, batch_size, input_via):

        pt_model = torch.load(f"Player-Data/{self._dataset}/mpc_model.pt")
        layers = ml.layers_from_torch(pt_model, input_shape, batch_size, input_via=input_via)

        model = ml.SGD(layers)

        return model