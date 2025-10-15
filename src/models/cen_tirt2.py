# In models/cen_tirt_hybrid.py

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Embedding
from tensorflow.keras.models import Model

class CEN:
    def __init__(self, inp_size_person_net, n_persons, n_trait, n_item, n_comps,
                 person_net_depth, show_model_layout):
        self.inp_size_person_net = inp_size_person_net
        self.n_persons, self.n_trait, self.n_item, self.n_comps = n_persons, n_trait, n_item, n_comps
        self.person_net_depth = person_net_depth
        self.show_model_layout = show_model_layout
        # Initialize attributes
        self.person_net, self.IRT_net, self.combined = None, None, None
        self.lambda_embedding, self.gamma_embedding = None, None
        self.res_mat, self.trait_id, self.item_id, self.reverse_id = None, None, None, None

    def _build_person_net(self):
        inp = Input(shape=(self.inp_size_person_net,), name="input_of_person_net")
        x = Dense(math.ceil(self.inp_size_person_net / 2), activation="relu")(inp)
        if self.person_net_depth == 3:
            x = Dense(math.ceil(self.inp_size_person_net / 4), activation="relu")(x)
            x = Dense(math.ceil(self.inp_size_person_net / 8), activation="relu")(x)
        eta_out = Dense(self.n_trait, activation="linear", name="eta_output")(x)
        eta_out_bn = BatchNormalization(momentum=0.99, epsilon=1e-6, center=False, scale=False)(eta_out)
        return Model(inp, eta_out_bn, name="person_net")

    def _build_IRT_net(self):
        inp_eta = Input(shape=(self.n_trait,), name="input_of_IRT_net_person")
        inp_lambda = Input(shape=(self.n_item,), name="input_of_IRT_net_lambda")
        inp_gamma = Input(shape=(self.n_comps,), name="input_of_IRT_net_gamma")
        inp_indices = Input(shape=(5,), dtype=tf.int32, name="input_of_IRT_net_indices")

        def compute_prob(inputs):
            eta, lambda_all, gamma_all, indices = inputs
            b_idx = tf.range(tf.shape(eta)[0], dtype=tf.int32)
            t_a, t_b, i_i, i_k, c_idx = indices[:,0], indices[:,1], indices[:,2], indices[:,3], indices[:,4]
            
            # Keras 3 requires explicit casting for gather indices
            eta_a = tf.gather(eta, tf.cast(t_a, dtype=tf.int32), batch_dims=1)
            eta_b = tf.gather(eta, tf.cast(t_b, dtype=tf.int32), batch_dims=1)
            lambda_i = tf.gather(lambda_all, tf.cast(i_i, dtype=tf.int32), batch_dims=1)
            lambda_k = tf.gather(lambda_all, tf.cast(i_k, dtype=tf.int32), batch_dims=1)
            gamma_l = tf.gather(gamma_all, tf.cast(c_idx, dtype=tf.int32), batch_dims=1)
            
            # Using fixed psi_sq = 1 constraint
            psi_sq_i, psi_sq_k = 1.0, 1.0 
            
            num = -gamma_l + lambda_i * eta_a - lambda_k * eta_b
            den = tf.sqrt(psi_sq_i + psi_sq_k + 1e-8)
            prob = tf.sigmoid(1.702 * (num / den))
            return prob

        prob = Lambda(compute_prob, output_shape=(1,))([inp_eta, inp_lambda, inp_gamma, inp_indices])
        return Model([inp_eta, inp_lambda, inp_gamma, inp_indices], prob, name="IRT_net")

    def load_data(self, res_mat, trait_id, item_id, reverse_id):
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
        
        t_ext, i_ext = np.tile(self.trait_id, (n_persons, 1)), np.tile(self.item_id, (n_persons, 1))
        c_ext = np.tile(np.arange(self.n_comps), n_persons).reshape(-1, 1)
        self.X_indices = np.hstack([t_ext, i_ext, c_ext])

    def build_networks(self):
        self._preprocess_data()
        self.person_net = self._build_person_net()
        self.IRT_net = self._build_IRT_net()
        
        self.lambda_embedding = Embedding(input_dim=1, output_dim=self.n_item, name="lambda_embedding")
        self.gamma_embedding = Embedding(input_dim=1, output_dim=self.n_comps, name="gamma_embedding")

        in_p = Input(shape=(self.inp_size_person_net,), name="combined_inp_person")
        in_idx = Input(shape=(5,), dtype='int32', name="combined_inp_indices")
        dummy_input = Input(shape=(1,), dtype='int32', name="dummy_input")

        eta_out = self.person_net(in_p)
        
        # CRITICAL FIX: Wrap tf.squeeze in Lambda layers
        lambda_out_raw = self.lambda_embedding(dummy_input)
        lambda_out = Lambda(lambda x: tf.squeeze(x, axis=1))(lambda_out_raw)
        
        gamma_out_raw = self.gamma_embedding(dummy_input)
        gamma_out = Lambda(lambda x: tf.squeeze(x, axis=1))(gamma_out_raw)
        
        prob = self.IRT_net([eta_out, lambda_out, gamma_out, in_idx])
        self.combined = Model([in_p, in_idx, dummy_input], prob, name="hybrid_network")
        if self.show_model_layout: self.combined.summary()

    def train(self, optimizer, loss_func, epochs, batch_size, early_stopping=None, verbose=1):
        dummy_data = np.zeros((self.X_person_net.shape[0], 1), dtype=np.int32)
        
        self.combined.compile(optimizer=optimizer, loss=loss_func)
        self.history = self.combined.fit(
            x=[self.X_person_net, self.X_indices, dummy_data], y=self.y_CEN,
            epochs=epochs, batch_size=batch_size, callbacks=[early_stopping] if early_stopping else [],
            verbose=verbose, validation_split=0.1)

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
        lambda_est = self.lambda_embedding.get_weights()[0][0]
        gamma_est = self.gamma_embedding.get_weights()[0][0]
        psi_sq_est = np.ones(self.n_item)
        
        return {"eta": eta_est, "lambda": lambda_est, "psi_sq": psi_sq_est, "gamma": gamma_est}