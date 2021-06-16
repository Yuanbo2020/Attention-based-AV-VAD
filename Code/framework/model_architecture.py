#coding= utf-8
import numpy as np
from keras.layers import Input, merge, Reshape, Dense
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import *
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.models import Model, load_model
from keras import layers
from keras.optimizers import Adam
import framework.config as config
from keras import backend as K

np.random.seed(config.random_seed)


def joint_independent_compile(model):
    model.compile(loss={
        'audio_singing_output': 'binary_crossentropy',
        'audio_speech_output': 'binary_crossentropy',
        'audio_silence_output': 'binary_crossentropy',
        'audio_others_output': 'binary_crossentropy',

        'video_open_output': 'binary_crossentropy',
        'video_close_output': 'binary_crossentropy',

        'joint_singing_out': 'binary_crossentropy',
        'joint_speech_out': 'binary_crossentropy',
        'joint_silence_out': 'binary_crossentropy',
        'joint_others_out': 'binary_crossentropy'},
        loss_weights={
            'audio_singing_output': 1.0,
            'audio_speech_output': 1.0,
            'audio_silence_output': 1.0,
            'audio_others_output': 1.0,

            'video_open_output': 1.0,
            'video_close_output': 1.0,

            'joint_singing_out': 1.0,
            'joint_speech_out': 1.0,
            'joint_silence_out': 1.0,
            'joint_others_out': 1.0},
        optimizer='adam', metrics=['accuracy'])


def joint_independent_compile_attention(model):
    model.compile(loss={
        'audio_singing_output': 'binary_crossentropy',
        'audio_speech_output': 'binary_crossentropy',
        'audio_silence_output': 'binary_crossentropy',
        'audio_others_output': 'binary_crossentropy',

        'video_final_out': 'binary_crossentropy',

        'joint_singing_out': 'binary_crossentropy',
        'joint_speech_out': 'binary_crossentropy',
        'joint_silence_out': 'binary_crossentropy',
        'joint_others_out': 'binary_crossentropy'},
        loss_weights={
            'audio_singing_output': 1.0,
            'audio_speech_output': 1.0,
            'audio_silence_output': 1.0,
            'audio_others_output': 1.0,

            'video_final_out': 1.0,

            'joint_singing_out': 1.0,
            'joint_speech_out': 1.0,
            'joint_silence_out': 1.0,
            'joint_others_out': 1.0},
        optimizer='adam', metrics=['accuracy'])


def audio_independent_compile(model):
    model.compile(loss={
        'audio_singing_output': 'binary_crossentropy',
        'audio_speech_output': 'binary_crossentropy',
        'audio_silence_output': 'binary_crossentropy',
        'audio_others_output': 'binary_crossentropy'},
        loss_weights={
            'audio_singing_output': 1.0,
            'audio_speech_output': 1.0,
            'audio_silence_output': 1.0,
            'audio_others_output': 1.0},
        optimizer='adam', metrics=['accuracy'])


def audio_pooling_layer(audio_c):
    audio_p = layers.Conv2D(32, (3, 3), strides=(1, 2), padding='same')(audio_c)  # (None, 15, 32, 32)
    audio_p = layers.BatchNormalization()(audio_p)
    audio_p = layers.ReLU()(audio_p)
    return audio_p


def audio_cnn_layer1(audio_input):
    audio_c1_l = layers.Conv2D(32, (3, 7), padding='same')(audio_input)  # (N, 15, 64, 32)
    audio_c1_l = layers.BatchNormalization()(audio_c1_l)
    audio_c1_l = Activation('linear')(audio_c1_l)

    audio_c1_s = layers.Conv2D(32, (3, 7), padding='same')(audio_input)  # (N, 15, 64, 32)
    audio_c1_s = layers.BatchNormalization()(audio_c1_s)
    audio_c1_s = Activation('sigmoid')(audio_c1_s)

    audio_c1 = Multiply()([audio_c1_l, audio_c1_s])
    audio_p1 = audio_pooling_layer(audio_c1)
    return audio_p1


def audio_cnn_layer2(audio_p1):
    audio_c2_l = layers.Conv2D(32, (3, 7), padding='same')(audio_p1)  # (N, 15, 32, 32)
    audio_c2_l = layers.BatchNormalization()(audio_c2_l)
    audio_c2_l = Activation('linear')(audio_c2_l)

    audio_c2_s = layers.Conv2D(32, (3, 7), padding='same')(audio_p1)  # (N, 15, 32, 32)
    audio_c2_s = layers.BatchNormalization()(audio_c2_s)
    audio_c2_s = Activation('sigmoid')(audio_c2_s)

    audio_c2 = Multiply()([audio_c2_l, audio_c2_s])  # (N, 15, 32, 32)
    audio_p2 = audio_pooling_layer(audio_c2)
    return audio_p2


def audio_cnn_layer3(audio_p2):
    audio_c3_l = layers.Conv2D(32, (3, 5), padding='same')(audio_p2)  # (N, 15, 16, 32)
    audio_c3_l = layers.BatchNormalization()(audio_c3_l)
    audio_c3_l = Activation('linear')(audio_c3_l)

    audio_c3_s = layers.Conv2D(32, (3, 5), padding='same')(audio_p2)  # (N, 15, 16, 32)
    audio_c3_s = layers.BatchNormalization()(audio_c3_s)
    audio_c3_s = Activation('sigmoid')(audio_c3_s)

    audio_c3 = Multiply()([audio_c3_l, audio_c3_s])  # (N, 15, 16, 32)
    audio_p3 = audio_pooling_layer(audio_c3)
    return audio_p3


def audio_cnn_layer4(audio_p3):
    audio_c4_l = layers.Conv2D(32, (3, 5), padding='same')(audio_p3)  # (N, 15, 8, 32)
    audio_c4_l = layers.BatchNormalization()(audio_c4_l)
    audio_c4_l = Activation('linear')(audio_c4_l)

    audio_c4_s = layers.Conv2D(32, (3, 5), padding='same')(audio_p3)  # (N, 15, 8, 32)
    audio_c4_s = layers.BatchNormalization()(audio_c4_s)
    audio_c4_s = Activation('sigmoid')(audio_c4_s)

    audio_c4 = Multiply()([audio_c4_l, audio_c4_s])  # (N, 15, 16, 32)
    audio_p4 = audio_pooling_layer(audio_c4)
    return audio_p4


def audio_middle_part(audio_part_in):
    audio_part_in = layers.BatchNormalization()(audio_part_in)
    audio_part_in = layers.ReLU()(audio_part_in)  # (None, 15, 4, 16)
    audio_part_in = layers.Conv2D(16, (5, 3), strides=(3, 1), padding='same')(audio_part_in)  # (None, 15, 4, 16)
    audio_part_in = layers.BatchNormalization()(audio_part_in)
    audio_part_in = layers.ReLU()(audio_part_in)  # (None, 5, 4, 16)
    audio_part_in = Flatten()(audio_part_in)  # 320
    return audio_part_in


def video_cnn_layer1(video_input):
    video_c1_l = layers.Conv2D(32, (5, 13), padding='same')(video_input)  # (N, 450, 300, 32)
    video_c1_l = layers.BatchNormalization()(video_c1_l)
    video_c1_l = Activation('linear')(video_c1_l)

    video_c1_s = layers.Conv2D(32, (5, 13), padding='same')(video_input)  # (N, 450, 300, 32)
    video_c1_s = layers.BatchNormalization()(video_c1_s)
    video_c1_s = Activation('sigmoid')(video_c1_s)

    video_c1 = Multiply()([video_c1_l, video_c1_s])

    video_p1 = layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same')(video_c1)  # (None, 150, 100, 32)
    video_p1 = layers.BatchNormalization()(video_p1)
    video_p1 = layers.ReLU()(video_p1)
    return video_p1


def video_cnn_layer2(video_p1):
    video_c2_l = layers.Conv2D(32, (5, 13), padding='same')(video_p1)
    video_c2_l = layers.BatchNormalization()(video_c2_l)
    video_c2_l = Activation('linear')(video_c2_l)

    video_c2_s = layers.Conv2D(32, (5, 13), padding='same')(video_p1)
    video_c2_s = layers.BatchNormalization()(video_c2_s)
    video_c2_s = Activation('sigmoid')(video_c2_s)

    video_c2 = Multiply()([video_c2_l, video_c2_s])  # (N, 15, 32, 32)

    video_p2 = layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same')(video_c2)  # (None, 75, 50, 32)
    video_p2 = layers.BatchNormalization()(video_p2)
    video_p2 = layers.ReLU()(video_p2)
    return video_p2


def video_cnn_layer3(video_p2):
    video_c3_l = layers.Conv2D(32, (3, 11), padding='same')(video_p2)  # (None, 75, 50, 32)
    video_c3_l = layers.BatchNormalization()(video_c3_l)
    video_c3_l = Activation('linear')(video_c3_l)

    video_c3_s = layers.Conv2D(32, (3, 11), padding='same')(video_p2)  # (None, 75, 50, 32)
    video_c3_s = layers.BatchNormalization()(video_c3_s)
    video_c3_s = Activation('sigmoid')(video_c3_s)

    video_c3 = Multiply()([video_c3_l, video_c3_s])  # (N, 15, 16, 32)

    video_p3 = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(video_c3)  # (None, 25, 17, 32)
    # 75+5-1 = 69/2 = 35
    video_p3 = layers.BatchNormalization()(video_p3)
    video_p3 = layers.ReLU()(video_p3)
    return video_p3


def video_cnn_layer4(video_p3):
    video_c4_l = layers.Conv2D(32, (3, 7), padding='same')(video_p3)  # (None, 35, 29, 32)
    video_c4_l = layers.BatchNormalization()(video_c4_l)
    video_c4_l = Activation('linear')(video_c4_l)

    video_c4_s = layers.Conv2D(32, (3, 7), padding='same')(video_p3)  # (None, 35, 29, 32)
    video_c4_s = layers.BatchNormalization()(video_c4_s)
    video_c4_s = Activation('sigmoid')(video_c4_s)

    video_c4 = Multiply()([video_c4_l, video_c4_s])  # (N, 15, 16, 32)

    video_p4 = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(video_c4)  # (None, 13, 9, 32)
    video_p4 = layers.BatchNormalization()(video_p4)
    video_p4 = layers.ReLU()(video_p4)
    return video_p4


def video_middle_part(video_part_in):
    video_part_in = layers.BatchNormalization()(video_part_in)
    video_part_in = layers.ReLU()(video_part_in)  # (None, 15, 4, 16)
    video_part_in = layers.Conv2D(16, (3, 3), strides=(2, 1), padding='same')(video_part_in)  # (None, 15, 4, 16)
    video_part_in = layers.BatchNormalization()(video_part_in)
    video_part_in = layers.ReLU()(video_part_in)  # (None, 5, 4, 16)
    video_part_in = Flatten()(video_part_in)  # 320
    return video_part_in


def average_pooling(video_close_glu, name):
    video_close_GAP_input = Reshape(target_shape=(int(video_close_glu.get_shape()[1]), 1))(video_close_glu)
    video_close_output = GlobalAveragePooling1D(name=name)(video_close_GAP_input)
    return video_close_output


def model_framework(audio, image, audio_time, audio_freq, video_height, video_width, video_input_frames_num):

    if audio:
        audio_input = Input(shape=(audio_time, audio_freq, 1), name='audio_in')  # (N, 15, 64)

        audio_p1 = audio_cnn_layer1(audio_input)
        audio_p2 = audio_cnn_layer2(audio_p1)
        audio_p3 = audio_cnn_layer3(audio_p2)
        audio_p4 = audio_cnn_layer4(audio_p3)

        # audio singing
        rnn_size = 64
        conv_shape = audio_p4.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_p4)
        a1_rnn = GRU(rnn_size, name='audio_singing_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_singing_rnn_linear')(a1_rnnout)
        a1_rnnout_gate = GRU(rnn_size, name='audio_singing_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_singing_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_singing = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_singing_s = Dense(128, name='audio_singing_embeddings')(audio_singing)  # (N, 128)
        audio_singing_s = layers.BatchNormalization()(audio_singing_s)
        audio_singing_embeddings = Activation('sigmoid')(audio_singing_s) # (N, 128) # 这个128维的_embeddings是要用到后面共享融合的
        audio_singing_output = Dense(1, activation='sigmoid', name='audio_singing_output')(audio_singing_embeddings)


        # audio speech
        rnn_size = 64
        conv_shape = audio_p4.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_p4)
        a1_rnn = GRU(rnn_size, name='audio_speech_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_speech_rnn_linear')(a1_rnnout)
        a1_rnnout_gate = GRU(rnn_size, name='audio_speech_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_speech_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_speech = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_speech_s = Dense(128, name='audio_speech_embeddings')(audio_speech)  # (N, 128)
        audio_speech_s = layers.BatchNormalization()(audio_speech_s)
        audio_speech_embeddings = Activation('sigmoid')(audio_speech_s)  # (N, 128) # 这个128维的_embeddings是要用到后面共享融合的
        audio_speech_output = Dense(1, activation='sigmoid', name='audio_speech_output')(audio_speech_embeddings)

        # audio others
        rnn_size = 64
        conv_shape = audio_p4.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_p4)
        a1_rnn = GRU(rnn_size, name='audio_others_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_others_rnn_linear')(a1_rnnout)
        a1_rnnout_gate = GRU(rnn_size, name='audio_others_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_others_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_others = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_others_s = Dense(128, name='audio_others_embeddings')(audio_others)  # (N, 128)
        audio_others_s = layers.BatchNormalization()(audio_others_s)
        audio_others_embeddings = Activation('sigmoid')(audio_others_s)  # (N, 128) # 这个128维的_embeddings是要用到后面共享融合的
        audio_others_output = Dense(1, activation='sigmoid', name='audio_others_output')(audio_others_embeddings)

        # audio silence
        rnn_size = 64
        conv_shape = audio_p4.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_p4)
        a1_rnn = GRU(rnn_size, name='audio_silence_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_silence_rnn_linear')(a1_rnnout)
        a1_rnnout_gate = GRU(rnn_size, name='audio_silence_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_silence_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_silence = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_silence_s = Dense(128, name='audio_silence_embeddings')(audio_silence)  # (N, 128)
        audio_silence_s = layers.BatchNormalization()(audio_silence_s)
        audio_silence_embeddings = Activation('sigmoid')(audio_silence_s)  # (N, 128) # 这个128维的_embeddings是要用到后面共享融合的
        audio_silence_output = Dense(1, activation='sigmoid', name='audio_silence_output')(audio_silence_embeddings)

    if image:
            video_input = Input(shape=(video_height, video_width, video_input_frames_num), name='video_in')

            video_p1 = video_cnn_layer1(video_input)
            video_p2 = video_cnn_layer2(video_p1)
            video_p3 = video_cnn_layer3(video_p2)
            video_p4 = video_cnn_layer4(video_p3)

            # video vocalizing
            video_vocalization = Flatten()(video_p4)

            video_vocalization = Dense(128, name='video_vocalization_embeddings')(video_vocalization)  # (N, 128)
            video_vocalization = layers.BatchNormalization()(video_vocalization)
            video_vocalization = Activation('sigmoid')(video_vocalization)  # (N, 128) # 这个128维的_embeddings是要用到后面共享融合的
            video_vocalization_output = Dense(1, activation='sigmoid', name='video_vocalization_output')(video_vocalization)

    if audio and image:
            audio_matrix = Concatenate()([audio_singing_embeddings, audio_speech_embeddings,
                                          audio_silence_embeddings, audio_others_embeddings])
            audio_matrix = Reshape(target_shape=(128, 4))(audio_matrix)
            shape = audio_matrix.get_shape()
            print('audio_matrix shape:', shape)
            # audio_matrix shape: (None, 128, 4)

            # video_matrix = Reshape(target_shape=(1, 128))(video_close_s_bn)

            q_dot_k = K.batch_dot(video_glu_concat, audio_matrix)
            # (N, 128) * (N, 128, 4) = (N, 4)
            # joint_final_out shape: (None, 4)
            print('q_dot_k shape:', q_dot_k.get_shape())
            # joint_final_out shape: (None, 4)

            attention = K.softmax(q_dot_k)
            print('attention shape:', attention.get_shape())
            # joint_final_out shape: (None, 4)

            attention_matrix = K.tile(K.expand_dims(attention, axis=1), (1, 128, 1))
            print('attention_matrix shape:', attention_matrix.get_shape())
            # attention_matrix shape: (None, 128, 4)

            # audio_matrix_4_128 = K.permute_dimensions(audio_matrix, (0, 2, 1))
            # print('audio_matrix_4_128 shape:', audio_matrix_4_128.get_shape())
            # # audio_matrix_4_128 shape: (None, 4, 128)

            attention_out = Multiply()([audio_matrix, attention_matrix])
            print('attention_out shape:', attention_out.get_shape())
            # attention_out shape: (None, 128, 4)

            singing_attention = Dense(128, name='singing_attention_128')(attention_out[:, :, 0])
            singing_attention = layers.BatchNormalization()(singing_attention)
            singing_attention = layers.ReLU()(singing_attention)
            joint_singing_out = Dense(1, activation='sigmoid', name='joint_singing_out')(singing_attention)

            speech_attention = Dense(128, name='speech_attention_128')(attention_out[:, :, 1])
            speech_attention = layers.BatchNormalization()(speech_attention)
            speech_attention = layers.ReLU()(speech_attention)
            joint_speech_out = Dense(1, activation='sigmoid', name='joint_speech_out')(speech_attention)

            silence_attention = Dense(128, name='silence_attention_128')(attention_out[:, :, 2])
            silence_attention = layers.BatchNormalization()(silence_attention)
            silence_attention = layers.ReLU()(silence_attention)
            joint_silence_out = Dense(1, activation='sigmoid', name='joint_silence_out')(silence_attention)

            others_attention = Dense(128, name='others_attention_128')(attention_out[:, :, 3])
            others_attention = layers.BatchNormalization()(others_attention)
            others_attention = layers.ReLU()(others_attention)
            joint_others_out = Dense(1, activation='sigmoid', name='joint_others_out')(others_attention)

            model = Model(inputs=[audio_input, video_input],
                          outputs=[audio_singing_output, audio_speech_output,
                                   audio_silence_output, audio_others_output,
                                   video_vocalization_output,
                                   joint_singing_out, joint_speech_out, joint_silence_out, joint_others_out])
            model.summary()
            joint_independent_compile_attention(model)
            return model
