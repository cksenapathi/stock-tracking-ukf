from model import Model
import tensorflow as tf
import numpy as np


class Optimizer:
    def __init__(self, lr=.001):
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.lr = lr

    def loss(self, model: Model, x, y, tra=False):
        y_ = model.predict(x, training=tra)
        return self.loss_object(y_true=y, y_pred=y_)

    def grad(self, model: Model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, tra=True)
        grad = tape.gradient(loss_value, model.get_trainable_variables())
        return loss_value, grad

    def simple_grad(self, model: Model, pred, target):
        with tf.GradientTape() as tape:
            loss_value = self.loss_object(y_true=target, y_pred=pred)
        grad = tape.gradient(loss_value, model.get_trainable_variables())
        return loss_value, grad
