



def get_prediction(sentence):
    tokens = encode_sentence(sentence)
    inputs = tf.expand_dims(tokens, 0)

    output = Dcnn(inputs, training=False)

    sentiment = math.floor(output*2)

    if sentiment == 0:
        print("Model output :  {}\nPredicted sentiment : Negative.".format(
            output))
    elif sentiment == 1:
        print("Model output : {}\nPredicted sentiment : Positive.".format(
            output))

