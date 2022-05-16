import os
import pandas as pd
import numpy as np
import librosa
import librosa.display

import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io, transform
from mpl_toolkits.mplot3d import Axes3D


def cosine_similarity(u, v):
    '''Calculate the similarity between 1D arrays'''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def similarity_matrix(array):
    '''Calculate the similarity matrix by given a 2D array'''
    shape = array.shape
    similarity = np.zeros((shape[0], shape[0]))

    for i in range(shape[0]):
        for k in range(shape[0]):
            similarity[i][k] = cosine_similarity(array[i], array[k])

    return similarity


def load_text_file(text_path, speaker='Participant'):
    '''load transcript file and extract the text of the given speaker'''

    def tokenize_corpus(corpus):
        '''tokenzie a given list of string into list of words'''
        tokens = [x.split() for x in corpus]
        return tokens

    # only 'Ellie', 'Participant', 'both' are allow
    assert speaker in ['Ellie', 'Participant', 'both'], \
        "Argument --speaker could only be ['Ellie', 'Participant', 'both']"

    text_file = pd.read_csv(text_path)
    # tokenize the text file, filter out all \t space and unnecessary columns such as time, participent
    tokenized_words = tokenize_corpus(text_file.values.tolist()[i][0] for i in range(text_file.shape[0]))

    sentences = []
    sentences_idx = []

    if speaker == 'Ellie':
        for idx, sentence in enumerate(tokenized_words):
            if sentence[2] == 'Ellie':
                sentences.append(sentence[3:])
                sentences_idx.append(idx)
    elif speaker == 'Participant':
        for idx, sentence in enumerate(tokenized_words):
            if sentence[2] == 'Participant':
                sentences.append(sentence[3:])
                sentences_idx.append(idx)

    else:  # speaker == 'both'
        sentences = [tokenized_words[i][3:] for i in range(len(tokenized_words))]
        sentences_idx = list(range(len(tokenized_words)))

    # recombine 2D list of words into 1D list of sentence
    final_sentences = [" ".join(sentences[i]).lower() for i in range(len(sentences))]

    return final_sentences


def find_max_length(root_dir):
    '''find out the maximum lenghth of each features among all patients'''

    # initialize each value
    max_length = {'landmarks': 0,
                  'gaze_samples': 0,
                  'sentences': 0}

    for name in os.listdir(root_dir):
        name_path = os.path.join(root_dir, name)
        if os.path.isdir(name_path) and name.endswith('_P'):
            session = name.split('_')[0]
            print('searching through patient {} ...'.format(session))

            facial_landmarks_path = os.path.join(name_path, '{}_CLNF_features3D.txt'.format(session))
            gaze_direction_path = os.path.join(name_path, '{}_CLNF_gaze.txt'.format(session))
            text_path = os.path.join(name_path, '{}_TRANSCRIPT.csv'.format(session))

            facial_landmarks = pd.read_csv(facial_landmarks_path)
            if len(facial_landmarks) > max_length['landmarks']:
                max_length['landmarks'] = len(facial_landmarks)

            gaze_direction = pd.read_csv(gaze_direction_path)
            if len(gaze_direction) > max_length['gaze_samples']:
                max_length['gaze_samples'] = len(gaze_direction)

            sentences = load_text_file(text_path, speaker='Participant')
            if len(sentences) > max_length['sentences']:
                max_length['sentences'] = len(sentences)

    if max_length['gaze_samples'] != max_length['landmarks']:
        max_length['gaze_samples'] = max_length['landmarks']

    return max_length


########################################################################################################################
# plot or diagram related
########################################################################################################################

def show_spectrogram(audio_feature, audio_parameters, y_axis="log"):
    """Show log-spectrogram for a batch of samples.
    Arguments:
        audio_feature: 2D numpy.ndarray, extracted audio feature (spectra) in dB
        audio_parameters: dict, all parameters setting of STFT
                          we used for feature extraction
        y_axis: certain string, scale of the y axis. could be 'linear' or 'log'
    Return:
        plot the spectrogram
    """

    # transpose, so the column corresponds to time series
    audio_feature = np.transpose(audio_feature)

    plt.figure(figsize=(25, 10))
    im = librosa.display.specshow(audio_feature,
                                  sr=audio_parameters['sample_rate'],
                                  hop_length=audio_parameters['hop_size'],
                                  x_axis="time",
                                  y_axis=y_axis)
    plt.colorbar(format="%+2.f dB")
    return im


def show_mel_filter_banks(filter_banks, audio_parameters):
    """Show Mel filter bank for a batch of samples.
    Arguments:
        filter_banks: 2D numpy.ndarray, please use self.filter_banks to get the value,
                      but make sure load_audio(spectro_type='mel_spectrogram') is called
        audio_parameters: dict, all parameters setting of STFT
                                we used for feature extraction
    Return:
        visualize the mel filter banks
    """
    plt.figure(figsize=(25, 10))
    im = librosa.display.specshow(filter_banks,
                                  sr=audio_parameters['sample_rate'],
                                  x_axis="linear")
    plt.colorbar(format="%+2.f")
    return im


def show_text_correlation(text_feature, start_sent, sent_len):
    """Show the correlation between each sentence.
    Arguments:
        text_feature: dict, one attribute of DepressionDataset, which
                      includes converted sentence embedding vectors (2D numpy.ndarray)
        start_sent: int, start index of the sentence you want
        sent_len: int, number of sentence you want to compare
                  (size of correlation matrix)
    Return:
        plot the correlation matrix between sentences
    """
    # calculate correlation matrix
    correlation = np.corrcoef(text_feature['sentence_embeddings'][int(start_sent):int(start_sent + sent_len)])
    plt.figure(figsize=(12, 12))
    # plot heatmap
    heatmap = sns.heatmap(correlation, annot=True, fmt='.2g')  # cbar_kws={'label': 'correlation'}
    # set scale label
    heatmap.set_xticklabels(text_feature['indices'][int(start_sent):int(start_sent + sent_len)])  # rotation=-30
    heatmap.set_yticklabels(text_feature['indices'][int(start_sent):int(start_sent + sent_len)], rotation=0)
    # set label
    plt.xlabel("sentence number in conversation")
    plt.ylabel("sentence number in conversation")
    plt.show()


def show_similarity_matrix(text_feature, start_sent, sent_len):
    '''plot the result of similarity matrix as heatmap'''
    # calculate similarity
    similarity = similarity_matrix(text_feature['sentence_embeddings'][int(start_sent):int(start_sent + sent_len)])
    # plot heatmap
    plt.figure(figsize=(16, 16))
    heatmap = sns.heatmap(similarity, annot=True, fmt='.2g')  # cbar_kws={'label': 'correlation'}
    # set scale label
    heatmap.set_xticklabels(text_feature['indices'][int(start_sent):int(start_sent + sent_len)])  # rotation=-30
    heatmap.set_yticklabels(text_feature['indices'][int(start_sent):int(start_sent + sent_len)], rotation=0)
    # set label
    plt.xlabel("sentence number in conversation")
    plt.ylabel("sentence number in conversation")
    plt.show()