import tensorflow as tf


class MyCustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, ckpt_manager, checkpoint_path):
        self.ckpt_manager = ckpt_manager
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
        print(f"Checkpoint saved in {self.checkpoint_path}.")