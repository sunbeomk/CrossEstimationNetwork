# In models/cen_tirt.py

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Layer
from tensorflow.keras.models import Model
import time

class ConstraintLayer(Layer):
    """Custom Keras layer to fix marker psi_sq=1 and constrain marker lambdas."""
    def __init__(self, n_item, psi_sq_fixed_indices, lambda_positive_indices, **kwargs):
        super(ConstraintLayer, self).__init__(**kwargs)
        self.n_item, self.psi_sq_fixed_indices, self.lambda_positive_indices = \
            n_item, tf.constant(psi_sq_fixed_indices, dtype=tf.int32), tf.constant(lambda_positive_indices, dtype=tf.int32)

    def call(self, item_params):
        lambdas = item_params[:, :self.n_item]
        psi_sqs_raw = item_params[:, self.n_item:2*self.n_item]
        gammas = item_params[:, 2*self.n_item:]
        psi_sqs_positive = tf.math.softplus(psi_sqs_raw)
        psi_mask = tf.reduce_any(tf.equal(tf.range(self.n_item, dtype=tf.int32), tf.expand_dims(self.psi_sq_fixed_indices, 1)), axis=0)
        psi_sqs_constrained = tf.where(psi_mask, 1.0, psi_sqs_positive)
        lambda_mask = tf.reduce_any(tf.equal(tf.range(self.n_item, dtype=tf.int32), tf.expand_dims(self.lambda_positive_indices, 1)), axis=0)
        lambdas_constrained = tf.where(lambda_mask, tf.math.softplus(lambdas), lambdas)
        return tf.concat([lambdas_constrained, psi_sqs_constrained, gammas], axis=1)

    def get_config(self):
        config = super(ConstraintLayer, self).get_config()
        config.update({
            "n_item": self.n_item,
            "psi_sq_fixed_indices": self.psi_sq_fixed_indices.numpy().tolist(),
            "lambda_positive_indices": self.lambda_positive_indices.numpy().tolist(),
        })
        return config

class CEN:
    def __init__(
        self, inp_size_person_net, inp_size_item_net, n_trait, n_item, n_comps,
        person_net_depth, item_net_depth, psi_sq_fixed_indices,
        lambda_positive_indices, show_model_layout):
        
        self.inp_size_person_net = inp_size_person_net 
        self.inp_size_item_net = inp_size_item_net
        self.n_trait, self.n_item, self.n_comps = n_trait, n_item, n_comps
        self.person_net_depth, self.item_net_depth = person_net_depth, item_net_depth
        self.psi_sq_fixed_indices = psi_sq_fixed_indices
        self.lambda_positive_indices = lambda_positive_indices
        self.show_model_layout = show_model_layout
        self.person_net, self.item_net, self.IRT_net, self.combined = None, None, None, None
        self.res_mat, self.trait_id, self.item_id, self.reverse_id = None, None, None, None
        self.X_person_net, self.X_item_net, self.X_indices, self.y_CEN = None, None, None, None

    def _build_person_net(self):
        inp = Input(shape=(self.inp_size_person_net,), name="input_of_person_net")
        x = Dense(math.ceil(self.inp_size_person_net / 2), activation="relu")(inp)
        if self.person_net_depth == 3:
            x = Dense(math.ceil(self.inp_size_person_net / 4), activation="relu")(x)
            x = Dense(math.ceil(self.inp_size_person_net / 8), activation="relu")(x)
        eta_out = Dense(self.n_trait, activation="linear", name="eta_output")(x)
        eta_out_bn = BatchNormalization(momentum=0.99, epsilon=1e-6, center=False, scale=False)(eta_out)
        return Model(inp, eta_out_bn, name="person_net")

    def _build_item_net(self):
        inp = Input(shape=(self.inp_size_item_net,), name="input_of_item_net")
        x = Dense(math.ceil(self.inp_size_item_net / 2), activation="relu")(inp)
        if self.item_net_depth == 3:
            x = Dense(math.ceil(self.inp_size_item_net / 4), activation="relu")(x)
            x = Dense(math.ceil(self.inp_size_item_net / 8), activation="relu")(x)
        params_out = Dense(2 * self.n_item + self.n_comps, activation="linear")(x)
        constrained_params = ConstraintLayer(
            n_item=self.n_item,
            psi_sq_fixed_indices=self.psi_sq_fixed_indices,
            lambda_positive_indices=self.lambda_positive_indices,
            name="constraints")(params_out)
        return Model(inp, constrained_params, name="item_net")
    
    def _build_IRT_net(self):
        inp_eta = Input(shape=(self.n_trait,), name="input_of_IRT_net_person")
        inp_params = Input(shape=(2 * self.n_item + self.n_comps,), name="input_of_IRT_net_item")
        inp_indices = Input(shape=(5,), dtype=tf.int32, name="input_of_IRT_net_indices")

        def compute_prob(inputs):
            eta, item_params, indices = inputs
            b_idx = tf.range(tf.shape(eta)[0], dtype=tf.int32)
            lambda_all, psi_sq_all, gamma_all = item_params[:,:self.n_item], item_params[:,self.n_item:2*self.n_item], item_params[:,2*self.n_item:]
            t_a, t_b, i_i, i_k, c_idx = indices[:,0], indices[:,1], indices[:,2], indices[:,3], indices[:,4]
            eta_a, eta_b = tf.gather_nd(eta, tf.stack([b_idx, t_a], 1)), tf.gather_nd(eta, tf.stack([b_idx, t_b], 1))
            lambda_i, lambda_k = tf.gather_nd(lambda_all, tf.stack([b_idx, i_i], 1)), tf.gather_nd(lambda_all, tf.stack([b_idx, i_k], 1))
            psi_sq_i, psi_sq_k = tf.gather_nd(psi_sq_all, tf.stack([b_idx, i_i], 1)), tf.gather_nd(psi_sq_all, tf.stack([b_idx, i_k], 1))
            gamma_l = tf.gather_nd(gamma_all, tf.stack([b_idx, c_idx], 1))
            num = -gamma_l + lambda_i * eta_a - lambda_k * eta_b
            den = tf.sqrt(psi_sq_i + psi_sq_k + 1e-8)
            z = num / den
            prob = tf.sigmoid(1.702 * z)
            return prob

        prob = Lambda(compute_prob, output_shape=(1,))([inp_eta, inp_params, inp_indices])
        return Model([inp_eta, inp_params, inp_indices], prob, name="IRT_net")

    # CRITICAL FIX: Add 'reverse_id' to the method signature
    def load_data(self, res_mat, trait_id, item_id, reverse_id):
        self.n_persons, _ = res_mat.shape
        self.res_mat, self.trait_id, self.item_id, self.reverse_id = res_mat, trait_id, item_id, reverse_id

    def _preprocess_data(self):
        n_persons, n_comps = self.res_mat.shape
        reverse_id_vec = self.reverse_id.astype(bool).flatten()
        ipsative_scores = np.zeros((n_persons, self.n_trait), dtype=int)
        
        for i in range(n_persons):
            res_vec = self.res_mat[i, :]
            winning_traits_normal = self.trait_id[np.arange(n_comps), 1 - res_vec]
            winning_traits_reversed = self.trait_id[np.arange(n_comps), res_vec]
            all_winning_traits = np.where(reverse_id_vec, winning_traits_reversed, winning_traits_normal)
            counts = np.bincount(all_winning_traits, minlength=self.n_trait)
            ipsative_scores[i, :] = counts
            
        self.y_CEN = self.res_mat.flatten()
        original_X_person_net = np.repeat(self.res_mat, self.n_comps, axis=0)
        repeated_ipsative_scores = np.repeat(ipsative_scores, self.n_comps, axis=0)
        self.X_person_net = np.hstack([original_X_person_net, repeated_ipsative_scores])
        
        self.X_item_net = np.tile(self.res_mat.T, (self.n_persons, 1))
        t_ext, i_ext = np.tile(self.trait_id, (self.n_persons, 1)), np.tile(self.item_id, (self.n_persons, 1))
        c_ext = np.tile(np.arange(self.n_comps), self.n_persons).reshape(-1, 1)
        self.X_indices = np.hstack([t_ext, i_ext, c_ext])

    def build_networks(self):
        self._preprocess_data()
        self.person_net, self.item_net, self.IRT_net = self._build_person_net(), self._build_item_net(), self._build_IRT_net()
        in_p = Input(shape=(self.inp_size_person_net,), name="combined_inp_person")
        in_i = Input(shape=(self.inp_size_item_net,), name="combined_inp_item")
        in_idx = Input(shape=(5,), dtype='int32', name="combined_inp_indices")
        eta_out, item_out = self.person_net(in_p), self.item_net(in_i)
        prob = self.IRT_net([eta_out, item_out, in_idx])
        self.combined = Model([in_p, in_i, in_idx], prob, name="cross_estimation_network")
        if self.show_model_layout: self.combined.summary()

    def train(self, optimizer, loss_func, epochs, batch_size, early_stopping=None, verbose=1):
        self.combined.compile(optimizer=optimizer, loss=loss_func)
        self.history = self.combined.fit(
            x=[self.X_person_net, self.X_item_net, self.X_indices], y=self.y_CEN,
            epochs=epochs, batch_size=batch_size, callbacks=[early_stopping] if early_stopping else [],
            verbose=verbose, validation_split=0.1)

    # CRITICAL FIX: Update param_est to handle ipsative scores and softplus
    def param_est(self):
        n_persons, n_comps = self.res_mat.shape
        reverse_id_vec = self.reverse_id.astype(bool).flatten()
        ipsative_scores = np.zeros((n_persons, self.n_trait), dtype=int)
        for i in range(n_persons):
            res_vec = self.res_mat[i, :]
            winning_traits_normal = self.trait_id[np.arange(n_comps), 1 - res_vec]
            winning_traits_reversed = self.trait_id[np.arange(n_comps), res_vec]
            all_winning_traits = np.where(reverse_id_vec, winning_traits_reversed, winning_traits_normal)
            counts = np.bincount(all_winning_traits, minlength=self.n_trait)
            ipsative_scores[i, :] = counts
        person_net_input = np.hstack([self.res_mat, ipsative_scores])
        
        eta_est = self.person_net.predict(person_net_input, verbose=0)
        item_params = self.item_net.predict(self.res_mat.T, verbose=0)
        
        lambda_s, psi_sq_s = np.zeros(self.n_item), np.zeros(self.n_item)
        item_c = np.zeros(self.n_item, dtype=int)
        
        all_l = item_params[:, :self.n_item]
        all_psi_sq = item_params[:, self.n_item:2*self.n_item] # Already final values from ConstraintLayer
        all_g = item_params[:, 2*self.n_item:]
        
        for c_idx in range(self.n_comps):
            i, k = self.item_id[c_idx]
            lambda_s[i] += all_l[c_idx, i]; item_c[i] += 1
            lambda_s[k] += all_l[c_idx, k]; item_c[k] += 1
            psi_sq_s[i] += all_psi_sq[c_idx, i]
            psi_sq_s[k] += all_psi_sq[c_idx, k]
        
        lambda_est = np.where(item_c > 0, lambda_s / item_c, 0)
        psi_sq_est = np.where(item_c > 0, psi_sq_s / item_c, 0)
        gamma_est = np.diag(all_g)
        
        return {"eta": eta_est, "lambda": lambda_est, "psi_sq": psi_sq_est, "gamma": gamma_est}