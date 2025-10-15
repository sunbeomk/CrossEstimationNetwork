#!/usr/bin/env python3
"""Run the empirical study with the Hybrid CEN model."""

import sys
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen_tirt2 import CEN

# ===============================
# Paths to data files
# ===============================
path_emp_data = "./data/tirt_data.csv"
path_trait_id = "./data/trait_id.csv"
path_item_id = "./data/item_id.csv"
path_reverse_id = "./data/reverse_id.csv"

# Load and preprocess data
res_mat = np.loadtxt(path_emp_data, skiprows=1, dtype=int, delimiter=",")
trait_id = np.loadtxt(path_trait_id, skiprows=1, dtype=int, delimiter=",") - 1
item_id = np.loadtxt(path_item_id, skiprows=1, dtype=int, delimiter=",") - 1
reverse_id_int = np.loadtxt(path_reverse_id, skiprows=1, dtype=int, delimiter=",")
reverse_id = reverse_id_int.astype(bool)

n_trait = len(np.unique(trait_id))
n_item = len(np.unique(item_id))

# ===============================
# Initialize CEN object (Hybrid version)
# ===============================
cen = CEN(
    inp_size_person_net=res_mat.shape[1] + n_trait,
    n_persons=res_mat.shape[0],
    n_trait=n_trait,
    n_item=n_item,
    n_comps=res_mat.shape[1],
    person_net_depth=3,
    show_model_layout=True,
)

cen.load_data(
    res_mat=res_mat,
    trait_id=trait_id,
    item_id=item_id,
    reverse_id=reverse_id
)

cen.build_networks()

# ===============================
# Set optimizer, loss, callbacks
# ===============================
optimizer = Adam(learning_rate=0.0005)
loss_func = BinaryCrossentropy()
early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.0001, patience=150,
    mode="min", restore_best_weights=True)

# ===============================
# Train the model
# ===============================
cen.train(
    optimizer=optimizer, loss_func=loss_func, epochs=10000,
    batch_size=res_mat.size, early_stopping=early_stopping, verbose=2)

# ===============================
# Parameter estimation & Saving
# ===============================
print("Extracting and saving final parameter estimates...")
estimates = cen.param_est()
path_emp_est = "./results/emp_study_hybrid"
os.makedirs(path_emp_est, exist_ok=True)
for name, params in estimates.items():
    np.savetxt(os.path.join(path_emp_est, f"{name}_est_cen.csv"), params, delimiter=",")

print("✅ Hybrid CEN analysis completed successfully.")