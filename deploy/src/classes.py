import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self, nbr_TamanioVoc_Input, nbr_EmbeddingDim, nbr_EncDec_Units, nbr_TamanioBatch):
        super(Encoder, self).__init__()
        self.nbr_TamanioBatch = nbr_TamanioBatch
        self.nbr_EncDec_Units = nbr_EncDec_Units
        self.embedding = tf.keras.layers.Embedding(nbr_TamanioVoc_Input, nbr_EmbeddingDim)
        self.gru = tf.keras.layers.GRU(self.nbr_EncDec_Units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tensor_Input_Batch, tensor_Encoder_Oculto):

        tensor_Input_Batch_Embedding = self.embedding(tensor_Input_Batch)

        tensor_Encoder_Output, tensor_Encoder_Estado_Oculto = self.gru(tensor_Input_Batch_Embedding, initial_state = tensor_Encoder_Oculto)
        return tensor_Encoder_Output, tensor_Encoder_Estado_Oculto

    def initialize_hidden_state(self):
        return tf.zeros((self.nbr_TamanioBatch, self.nbr_EncDec_Units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, tensor_Oculto, tensor_Encoder_Output):
        # tensor_Oculto hidden state shape == (batch_size, hidden size)
        # tensor_Oculto_with_time_axis shape == (batch_size, 1, hidden size)
        # tensor_Encoder_Output shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        tensor_Oculto_with_time_axis = tf.expand_dims(tensor_Oculto, 1)
        # print('tensor_Oculto: ', tensor_Oculto) # MJMA
        # print('tensor_Oculto_with_time_axis: ', tensor_Oculto_with_time_axis) # MJMA

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
                                  self.W1(tensor_Oculto_with_time_axis) + self.W2(tensor_Encoder_Output)
                                  )
                      )

        # tensor_attention_weights shape == (batch_size, max_length, 1)
        tensor_attention_weights = tf.nn.softmax(score, axis=1)

        # tensor_context_vector shape after sum == (batch_size, hidden_size)
        tensor_context_vector = tensor_attention_weights * tensor_Encoder_Output
        tensor_context_vector = tf.reduce_sum(tensor_context_vector, axis=1)

        return tensor_context_vector, tensor_attention_weights


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, nbr_EmbeddingDim, nbr_EncDec_Units, nbr_TamanioBatch):
        super(Decoder, self).__init__()
        self.nbr_TamanioBatch = nbr_TamanioBatch
        self.nbr_EncDec_Units = nbr_EncDec_Units
        self.embedding = tf.keras.layers.Embedding(vocab_size, nbr_EmbeddingDim)
        self.gru = tf.keras.layers.GRU(self.nbr_EncDec_Units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.nbr_EncDec_Units)

    def call(self, tensor_input, tensor_Oculto, tensor_Encoder_Output):

        # tensor_Encoder_Output shape == (batch_size, max_length, hidden_size)
        tensor_context_vector, tensor_attention_weights = self.attention.call(tensor_Oculto, tensor_Encoder_Output)


        # tensor_input shape after passing through embedding == (batch_size, 1, nbr_EmbeddingDim)
        tensor_input_embedding = self.embedding(tensor_input)

        # x shape after concatenation == (batch_size, 1, nbr_EmbeddingDim + hidden_size)
        tensor_input_embedding_concatenado = tf.concat([tf.expand_dims(tensor_context_vector, 1), tensor_input_embedding], axis=-1)

        # passing the concatenated vector to the GRU
        tensor_output, tensor_Decoder_Estado = self.gru(tensor_input_embedding_concatenado)

        # tensor_output shape == (batch_size * 1, hidden_size)
        tensor_output = tf.reshape(tensor_output, (-1, tensor_output.shape[2]))

        # tensor_output shape == (batch_size, vocab)
        tensor_Decoder_Output = self.fc(tensor_output)

        return tensor_Decoder_Output, tensor_Decoder_Estado, tensor_attention_weights
