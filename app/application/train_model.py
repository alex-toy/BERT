import app.config as cf
import tensorflow as tf
import time

from app.model.loss_function import loss_function


def train_model(dataset, ckpt_manager, transformer, optimizer) :
    for epoch in range(cf.EPOCHS):
            print("Start of epoch {}".format(epoch+1))
            start = time.time()
            
            train_loss = tf.keras.metrics.Mean(name="train_loss")
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

            train_loss.reset_states()
            train_accuracy.reset_states()
            
            for (batch, (enc_inputs, targets)) in enumerate(dataset):
                dec_inputs = targets[:, :-1]
                dec_outputs_real = targets[:, 1:]
                with tf.GradientTape() as tape:
                    predictions = transformer(enc_inputs, dec_inputs, True)
                    loss = loss_function(dec_outputs_real, predictions)
                
                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
                
                train_loss(loss)
                train_accuracy(dec_outputs_real, predictions)
                
                if batch % 50 == 0:
                    print("Epoch {} Batch {} Loss {:.4f} Precision {:.4f}".format(
                        epoch+1, batch, train_loss.result(), train_accuracy.result()))
                    
            ckpt_save_path = ckpt_manager.save()
            print("Keep checkpoint for epoch {} in {}".format(epoch+1, ckpt_save_path))
            print(f"Duration of epoch {epoch} : {time.time() - start} secs\n")

    return transformer