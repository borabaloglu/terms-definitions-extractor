# @author: felixhao28.

# region Import libraries
from keras.layers import Dense, Lambda, dot, Activation, concatenate
# endregion


def attention_3d_block(hidden_states):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name="attention_score_vec")(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name="last_hidden_state")(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name="attention_score")
    attention_weights = Activation("softmax", name="attention_weight")(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name="context_vector")
    pre_activation = concatenate([context_vector, h_t], name="attention_output")
    attention_vector = Dense(128, use_bias=False, activation="tanh", name="attention_vector")(pre_activation)
    return attention_vector
