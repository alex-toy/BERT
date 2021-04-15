import re
import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import app.config as cf

from joblib import dump, load


class CleanData :
    """
    cleans text data from raw file
    """

    def __init__(self, 
            input_data_path, 
            input_nb_prefix_path, 
            output_data_path, 
            output_nb_prefix_path,
            MAX_LENGTH, 
            BATCH_SIZE, 
            BUFFER_SIZE
        ) :
        self.input_data_path = input_data_path
        self.input_nb_prefix_path = input_nb_prefix_path
        self.output_data_path = output_data_path
        self.output_nb_prefix_path = output_nb_prefix_path
        self.MAX_LENGTH = MAX_LENGTH
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE


    def get_data(self):
        with open(self.input_data_path, mode = "r", encoding = "utf-8") as f:
            input_corpus = f.read()
        
        with open(self.output_data_path, mode = "r", encoding = "utf-8") as f:
            output_corpus = f.read()

        print('get_data')
        return input_corpus, output_corpus

    
    def get_non_breaking_prefix_data(self):
        with open(self.input_nb_prefix_path, mode = "r", encoding = "utf-8") as f:
            input_nb_prefix = f.read()
        
        with open(self.output_nb_prefix_path, mode = "r", encoding = "utf-8") as f:
            output_nb_prefix = f.read()

        print('get_non_breaking_prefix_data')
        return input_nb_prefix, output_nb_prefix


    def get_non_breaking_prefix(self) :
        input_nb_prefix, output_nb_prefix = self.get_non_breaking_prefix_data()
        
        non_breaking_prefix = input_nb_prefix.split("\n")
        input_non_breaking_prefix = [' ' + pref + '.' for pref in non_breaking_prefix]

        non_breaking_prefix = output_nb_prefix.split("\n")
        output_non_breaking_prefix = [' ' + pref + '.' for pref in non_breaking_prefix]
        
        print('get_non_breaking_prefix')
        return input_non_breaking_prefix, output_non_breaking_prefix


    def get_regex_cleaned_corpus(self, corpus) :
        corpus = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus)
        corpus = re.sub(r"\.\$\$\$", '', corpus)
        corpus = re.sub(r"  +", " ", corpus)
        print('get_regex_cleaned_corpus')
        return corpus.split('\n')

    
    def get_cleaned_corpus(self) :
        print('get_cleaned_corpus 1')
        input_corpus, output_corpus = self.get_data() 
        input_non_breaking_prefix, output_non_breaking_prefix = self.get_non_breaking_prefix()
        
        print('get_cleaned_corpus 2')
        for prefix in input_non_breaking_prefix:
            input_corpus = input_corpus.replace(prefix, prefix + '$$$')
        input_corpus = self.get_regex_cleaned_corpus(input_corpus)

        print('get_cleaned_corpus 3')
        for prefix in output_non_breaking_prefix:
            output_corpus = output_corpus.replace(prefix, prefix + '$$$')
        output_corpus = self.get_regex_cleaned_corpus(output_corpus)
        
        print('get_cleaned_corpus 4')
        return input_corpus, output_corpus
        

    def tokenize(self) :
        input_corpus, output_corpus = self.get_cleaned_corpus()
        st_enc = tfds.deprecated.text.SubwordTextEncoder
        print('tokenize')
        input_tokenizer = st_enc.build_from_corpus(
            input_corpus, target_vocab_size=2**13
        )
        print('Dump input_tokenizer')
        dump(input_tokenizer, os.path.join(cf.OUTPUTS_MODELS_DIR, 'input_tokenizer.joblib')) 
        
        output_tokenizer = st_enc.build_from_corpus(
            output_corpus, target_vocab_size=2**13
        )
        print('Dump output_tokenizer')
        dump(output_tokenizer, os.path.join(cf.OUTPUTS_MODELS_DIR, 'output_tokenizer.joblib')) 

        return input_tokenizer, output_tokenizer, input_corpus, output_corpus



    def get_puts(self) :
        input_tokenizer, output_tokenizer, input_corpus, output_corpus = self.tokenize()
        
        self.INPUT_VOCAB_SIZE = input_tokenizer.vocab_size + 2
        print('Dump  INPUT_VOCAB_SIZE')
        dump(self.INPUT_VOCAB_SIZE, os.path.join(cf.OUTPUTS_MODELS_DIR, 'INPUT_VOCAB_SIZE.joblib'))
        
        self.OUTPUT_VOCAB_SIZE = output_tokenizer.vocab_size + 2
        print('Dump  OUTPUT_VOCAB_SIZE')
        dump(self.OUTPUT_VOCAB_SIZE, os.path.join(cf.OUTPUTS_MODELS_DIR, 'OUTPUT_VOCAB_SIZE.joblib'))
        
        print('get_puts : inputs')
        inputs = [
            [self.INPUT_VOCAB_SIZE-2] + input_tokenizer.encode(sentence) + [self.INPUT_VOCAB_SIZE-1]
            for sentence in input_corpus
        ]
        print('get_puts : outputs')
        outputs = [
            [self.OUTPUT_VOCAB_SIZE-2] + output_tokenizer.encode(sentence) + [self.OUTPUT_VOCAB_SIZE-1]
            for sentence in input_corpus
        ]

        print('get_puts return')
        return inputs, outputs


    
    def remove_long_sentences(self) :
        inputs, outputs = self.get_puts() 
        print('remove_long_sentences 1')
        idx_to_remove = [count for count, sent in enumerate(inputs) if len(sent) > self.MAX_LENGTH]
        for idx in reversed(idx_to_remove):
            del inputs[idx]
            del outputs[idx]
        print('remove_long_sentences 2')
        idx_to_remove = [count for count, sent in enumerate(outputs) if len(sent) > self.MAX_LENGTH]
        for idx in reversed(idx_to_remove):
            del inputs[idx]
            del outputs[idx]
        print('remove_long_sentences 3')
        seq = tf.keras.preprocessing.sequence
        inputs = seq.pad_sequences(inputs, value=0, padding='post', maxlen=self.MAX_LENGTH)
        outputs = seq.pad_sequences(outputs, value=0, padding='post', maxlen=self.MAX_LENGTH)

        print('remove_long_sentences 4')
        return inputs, outputs


    def get_dataset(self) :
        inputs, outputs = self.remove_long_sentences()
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print('get_dataset')
        return dataset


    def path_to_csv(self) :
        inputs, outputs = self.remove_long_sentences()
        pd.DataFrame(inputs).to_csv(cf.INPUTS_FILE, index=False)
        pd.DataFrame(outputs).to_csv(cf.OUTPUTS_FILE, index=False)




if __name__ == "__main__":

    cd = CleanData(
        input_data_path=cf.europarl_en_path, 
        input_nb_prefix_path=cf.nb_prefix_en_path, 
        output_data_path=cf.europarl_es_path, 
        output_nb_prefix_path=cf.nb_prefix_es_path,
        MAX_LENGTH=20,
        BATCH_SIZE = 64,
        BUFFER_SIZE = 20000
    )

    dataset = cd.get_dataset()

    print(dataset)


    #cd.path_to_csv()


