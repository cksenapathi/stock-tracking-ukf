from model import Model
import tensorflow as tf
import numpy as np


class Optimizer:
    def __init__(self):
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        self._last_grad = []

    def loss(self, model: Model, x, y, tra=False):
        y_ = model.predict(x, training=tra)
        return self.loss_object(y_true=y, y_pred=y_)

    def grad(self, model: Model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, tra=True)
        curr_grad = tape.gradient(loss_value, model.get_trainable_variables())
        self._last_grad = curr_grad
        return loss_value, curr_grad

    def simple_grad(self, model: Model, pred, target):
        with tf.GradientTape() as tape:
            loss_value = self.loss_object(y_true=target, y_pred=pred)
        grad = tape.gradient(loss_value, model.get_trainable_variables())
        return loss_value, grad

    def get_weight_cov(self):
        weight_changes = np.array([])
        for i in self._last_grad:
            weight_changes = np.append(weight_changes, np.abs(i.numpy()))
        return np.outer(weight_changes - np.mean(weight_changes), weight_changes - np.mean(weight_changes))
