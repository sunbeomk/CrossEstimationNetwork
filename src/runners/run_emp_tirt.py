#!/usr/bin/env python3
"""Run the empirical study with the CEN model."""

import sys
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen_tirt import CEN

# ===============================
# Paths to data files
# ===============================
path_emp_data = "./data/tirt_data.csv"
path_trait_id = "./data/trait_id.csv"
path_item_id = "./data/item_id.csv"
path_block_id = "./data/block_id.csv"
path_reverse_id = "./data/reverse_id.csv"

# Load and preprocess data
res_mat = np.loadtxt(path_emp_data, skiprows=1, dtype=int, delimiter=",")
trait_id = np.loadtxt(path_trait_id, skiprows=1, dtype=int, delimiter=",") - 1
item_id = np.loadtxt(path_item_id, skiprows=1, dtype=int, delimiter=",") - 1
item_to_block_map = np.loadtxt(path_block_id, skiprows=1, dtype=int, delimiter=",") - 1

# Load reverse_id as a boolean vector
reverse_id_int = np.loadtxt(path_reverse_id, skiprows=1, dtype=int, delimiter=",")
reverse_id = reverse_id_int.astype(bool)

# ===============================
# Determine Indices for Constraints
# ===============================
print("--- Determining indices for model constraints ---")
unique_blocks = np.unique(item_to_block_map[:, 0])
psi_sq_fixed_indices = []
for block_idx in unique_blocks:
    items_in_block = item_to_block_map[item_to_block_map[:, 0] == block_idx, 1]
    last_item_in_block = np.max(items_in_block)
    psi_sq_fixed_indices.append(int(last_item_in_block))
psi_sq_fixed_indices = sorted(list(set(psi_sq_fixed_indices)))
print(f"Indices where psi_sq will be fixed to 1.0: {psi_sq_fixed_indices}")

n_trait = len(np.unique(trait_id))
n_item = len(np.unique(item_id))

# ===============================
# Initialize CEN object
# ===============================
cen = CEN(
    inp_size_person_net=res_mat.shape[1] + n_trait,
    inp_size_item_net=res_mat.shape[0],
    n_trait=n_trait,
    n_item=n_item,
    n_comps=res_mat.shape[1],
    person_net_depth=1,
    item_net_depth=1,
    psi_sq_fixed_indices=psi_sq_fixed_indices,
    reverse_id=reverse_id,
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
optimizer = Adam(learning_rate=0.001)
loss_func = BinaryCrossentropy()
early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.0001, patience=100,
    mode="min", restore_best_weights=True)


# ===============================
# Train the model
# ===============================
cen.train(
    optimizer=optimizer, 
    loss_func=loss_func, 
    epochs=10000,
    batch_size=res_mat.size, 
    early_stopping=early_stopping, 
    verbose=2,
)

# ===============================
# Parameter estimation & Saving
# ===============================
print("Extracting and saving final parameter estimates...")
estimates = cen.param_est()
path_emp_est = "./results/emp_study"
os.makedirs(path_emp_est, exist_ok=True)
for name, params in estimates.items():
    np.savetxt(os.path.join(path_emp_est, f"{name}_est_cen.csv"), params, delimiter=",")

print("✅ CEN analysis completed successfully.")