from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from model import Model


class Optimizer:
    def __init__(self):
        self.loss_object = tf.keras.losses.MeanSquaredError
        self.optimizer = tf.keras.optimizers.Adam(lr=.001)

    def loss(self, model, x, y, training=False):
        y_ = model.run(input=x, training=training)
        return self.loss_object(y, y_)

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.get_trainable_variables())

