from itertools import count
import os
import numpy as np
import pandas as pd
from skimage import transform

import torch
from torch.utils.data import Dataset

# local functions
from dataset.utils import *


class DepressionDataset(Dataset):
    '''create a training, develop, or test dataset
       and load the participant features if it's called
    '''

    def __init__(self,
                 root_dir,
                 mode,
                 use_mel_spectrogram=True,
                 visual_with_gaze=True,
                 transform=None):
        super(DepressionDataset, self).__init__()

        # # only train, develop, test dataset allow
        # assert mode in ["train", "validation", "test"], \
        #     "Argument --mode could only be ['train', 'validation', 'test']"
        
        self.mode = mode
        self.root_dir = root_dir
        self.use_mel_spectrogram = use_mel_spectrogram
        self.visual_with_gaze = visual_with_gaze
        self.transform = transform

        if mode == 'train':
            # assert mode in os.path.normpath(self.root_dir).split(os.path.sep), \
            #     "It seems like the given root_dir doesn't match the current mode '{}'".format(mode)

            # store ground truth
            self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'))
            self.gender_gt = np.load(os.path.join(self.root_dir, 'gender_gt.npy'))
            self.phq_binary_gt = np.load(os.path.join(self.root_dir, 'phq_binary_gt.npy'))
            self.phq_score_gt = np.load(os.path.join(self.root_dir, 'phq_score_gt.npy'))
            self.phq_subscores_gt = np.load(os.path.join(self.root_dir, 'phq_subscores_gt.npy'))
            
        elif mode == 'validation':
            # assert mode in os.path.normpath(self.root_dir).split(os.path.sep), \
            #     "It seems like the given root_dir doesn't match the current mode '{}'".format(mode)

            # store ground truth
            self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'))
            self.gender_gt = np.load(os.path.join(self.root_dir, 'gender_gt.npy'))
            self.phq_binary_gt = np.load(os.path.join(self.root_dir, 'phq_binary_gt.npy'))
            self.phq_score_gt = np.load(os.path.join(self.root_dir, 'phq_score_gt.npy'))
            self.phq_subscores_gt = np.load(os.path.join(self.root_dir, 'phq_subscores_gt.npy'))
        
        elif mode == 'test':
            # assert mode in os.path.normpath(self.root_dir).split(os.path.sep), \
            #     "It seems like the given root_dir doesn't match the current mode '{}'".format(mode)

            # store ground truth
            self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'))
            self.gender_gt = np.load(os.path.join(self.root_dir, 'gender_gt.npy'))
            self.phq_binary_gt = np.load(os.path.join(self.root_dir, 'phq_binary_gt.npy'))
            self.phq_score_gt = np.load(os.path.join(self.root_dir, 'phq_score_gt.npy'))


    def __len__(self):
        return len(self.IDs)

    def __iter__(self):
        return iter(self.IDs)

    def __getitem__(self, idx):
        '''
        Essentional function for creating dataset in PyTorch, which will automatically be
        called in Dataloader and load all the extracted features of the patient in the Batch
        based on the index of self.IDs
        Argument:
            idx: int, index of the patient ID in self.IDs
        Return:
            session: dict, contains all the extracted features and ground truth of a patient/session
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get visual feature
        if self.visual_with_gaze:
            fkps_path = os.path.join(self.root_dir, 'facial_keypoints')
            gaze_path = os.path.join(self.root_dir, 'gaze_vectors')

            # load and create final visual feature
            fkps_file = np.sort(os.listdir(fkps_path))[idx]
            gaze_file = np.sort(os.listdir(gaze_path))[idx]
            fkps = np.load(os.path.join(fkps_path, fkps_file))
            gaze = np.load(os.path.join(gaze_path, gaze_file))
            visual = np.concatenate((fkps, gaze), axis=1)
        else:
            fkps_path = os.path.join(self.root_dir, 'facial_keypoints')
            # load and create final visual feature
            fkps_file = np.sort(os.listdir(fkps_path))[idx]
            visual = np.load(os.path.join(fkps_path, fkps_file))

        # # get facial key points feature
        # kps_path = os.path.join(self.root_dir, 'facial_keypoints')
        # kps_file = np.sort(os.listdir(kps_path))[idx]
        # kps = np.load(os.path.join(kps_path, kps_file))

        # # get gaze vectors feature
        # gaze_path = os.path.join(self.root_dir, 'gaze_vectors')
        # gaze_file = np.sort(os.listdir(gaze_path))[idx]
        # gaze = np.load(os.path.join(gaze_path, gaze_file))

        # get audio feature
        if self.use_mel_spectrogram:
            audio_path = os.path.join(self.root_dir, 'audio', 'mel-spectrogram')
        else:
            audio_path = os.path.join(self.root_dir, 'audio', 'spectrogram')
        audio_file = np.sort(os.listdir(audio_path))[idx]
        audio = np.load(os.path.join(audio_path, audio_file))  # shape: frequency_bins x num_sample (80 x 1800)

        # get text feature
        text_path = os.path.join(self.root_dir, 'text')
        text_file = np.sort(os.listdir(text_path))[idx]
        text = np.load(os.path.join(text_path, text_file))

        # # get HOG feature
        # hog_path = os.path.join(self.root_dir, 'hog_features')
        # hog_file = np.sort(os.listdir(hog_path))[idx]
        # hog = np.load(os.path.join(hog_path, hog_file))

        # summary
        if self.mode == 'test':
            session = {'ID': self.IDs[idx],
                       'gender_gt': self.gender_gt[idx],
                       'phq_binary_gt': self.phq_binary_gt[idx],
                       'phq_score_gt': self.phq_score_gt[idx],
                       'visual': visual,
                       'audio': audio, 
                       'text': text}
        else:
            session = {'ID': self.IDs[idx],
                       'gender_gt': self.gender_gt[idx],
                       'phq_binary_gt': self.phq_binary_gt[idx],
                       'phq_score_gt': self.phq_score_gt[idx],
                       'phq_subscores_gt': self.phq_subscores_gt[idx],
                       'visual': visual,
                       'audio': audio, 
                       'text': text}

        if self.transform:
            session = self.transform(session)

        return session


class Padding(object):
    ''' pad zero to each feature matrix so that they all have the same size '''

    def __init__(self, audio_output_size=(80, 2000)):
        super(Padding, self).__init__()

        assert isinstance(audio_output_size, (int, tuple))
        self.audio_output_size = audio_output_size

    def __call__(self, session):
        padded_session = session
        audio = session['audio']

        # audio padding along width dimension
        if isinstance(self.audio_output_size, int):
            h, w = audio.shape
            new_w = self.audio_output_size if w > self.audio_output_size else w
            padded_audio = np.zeros((h, self.audio_output_size))
            padded_audio[:h, :new_w] = audio[:h, :new_w]

        # audio padding along both heigh and width dimension
        else:
            h, w = audio.shape
            new_h = self.audio_output_size[0] if h > self.audio_output_size[0] else h
            new_w = self.audio_output_size[1] if w > self.audio_output_size[1] else w
            padded_audio = np.zeros(self.audio_output_size)
            padded_audio[:new_h, :new_w] = audio[:new_h, :new_w]

        # summary
        padded_session['audio'] = padded_audio

        return padded_session


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Arguments:
        output_size:(tuple or int),  Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(80, 900)):
        assert isinstance(output_size, (int, tuple, list))

        if type(output_size) == list:
            assert len(output_size) == 2, "Rescale output size should be 2 dimensional"

        self.output_size = output_size

    def __call__(self, session):
        rescaled_session = session
        audio = session['audio']

        h, w = audio.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        rescaled_audio = transform.resize(audio, (new_h, new_w))

        # summary
        rescaled_session['audio'] = rescaled_audio

        return rescaled_session


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Arguments:
        output_size:(tuple or int), Desired output size. 
        If int, square crop is made.
    """

    def __init__(self, output_size=(224, 224)):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, session):
        cropped_session = session
        audio = session['audio']

        h, w = audio.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_audio = audio[top:top + new_h, left:left + new_w]

        # summary
        cropped_session['audio'] =  cropped_audio

        return cropped_session


class ToTensor(object):
    """Convert ndarrays in sample to Tensors or np.int to torch.tensor."""

    def __init__(self, mode):
        # assert mode in ["train", "validation", "test"], \
        #     "Argument --mode could only be ['train', 'validation', 'test']"
        
        self.mode = mode

    def __call__(self, session):
        if self.mode == 'test':
            converted_session = {'ID': session['ID'],
                                 'gender_gt': torch.tensor(session['gender_gt']).type(torch.FloatTensor),
                                 'phq_binary_gt': torch.tensor(session['phq_binary_gt']).type(torch.FloatTensor),
                                 'phq_score_gt': torch.tensor(session['phq_score_gt']).type(torch.FloatTensor),
                                 'visual': torch.from_numpy(session['visual']).type(torch.FloatTensor),
                                 'audio': torch.from_numpy(session['audio']).type(torch.FloatTensor),
                                 'text': torch.from_numpy(session['text']).type(torch.FloatTensor)}
        
        else:
            converted_session = {'ID': session['ID'],
                                 'gender_gt': torch.tensor(session['gender_gt']).type(torch.FloatTensor),
                                 'phq_binary_gt': torch.tensor(session['phq_binary_gt']).type(torch.FloatTensor),
                                 'phq_score_gt': torch.tensor(session['phq_score_gt']).type(torch.FloatTensor),
                                 'phq_subscores_gt': torch.from_numpy(session['phq_subscores_gt']).type(torch.FloatTensor),
                                 'visual': torch.from_numpy(session['visual']).type(torch.FloatTensor),
                                 'audio': torch.from_numpy(session['audio']).type(torch.FloatTensor),
                                 'text': torch.from_numpy(session['text']).type(torch.FloatTensor)}
        
        return converted_session


if __name__ == '__main__':
    from torch.utils.data import WeightedRandomSampler, DataLoader
    from torchvision import transforms

    root_dir = '/cvhci/temp/wpingcheng/DAIC_WOZ-generated_database_V2/train/clipped_data/'

    # test 3: try to load the dataset with DataLoader
    transformed_dataset = DepressionDataset(root_dir, 'train',
                                            transform=transforms.Compose([ToTensor('train')]))  # Rescale((80, 900)), Padding((90, 900)), 

    # show phq binary distribution
    phq_binary_gt = transformed_dataset.phq_binary_gt
    print('target train 0/1: {}/{}'.format(len(np.where(phq_binary_gt == 0)[0]), len(np.where(phq_binary_gt == 1)[0])))

    # show phq score distribution
    phq_score_gt = transformed_dataset.phq_score_gt
    class_sample_ID, class_sample_count = np.unique(phq_score_gt, return_counts=True)
    print('class_sample_ID   : {}'.format(class_sample_ID))
    print('class_sample_count: {}'.format(class_sample_count))
    print('='*90)

    weight = 1. / class_sample_count
    samples_weight = np.zeros(phq_score_gt.shape)
    for i, sample_id in enumerate(class_sample_ID):
        indices = np.where(phq_score_gt == sample_id)[0]
        value = weight[i]
        samples_weight[indices] = value

    # samples_weight = weight[phq_binary_gt]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # create dataloader
    dataloader = DataLoader(transformed_dataset,
                            batch_size=100,
                            num_workers=1,
                            sampler=sampler)


    # iterate through batches
    total_count = np.zeros(class_sample_ID.shape)
    for i_batch, sample_batched in enumerate(dataloader):
        print('Batch number: ', i_batch, ', audio: ', sample_batched['audio'].size())
        num_count = []
        for id in class_sample_ID:
            num_count.append(len(np.where(sample_batched['phq_score_gt'].numpy() == id)[0]))
        print('loaded data PHQ Score Classes     : {}'.format(class_sample_ID))
        print('loaded data PHQ Score Distribution: {}'.format(num_count))
        # print('Participant IDs: {}'.format(sample_batched['ID']))
        print('='*90)
        total_count += num_count
    print('Total chosen classes: {}'.format(class_sample_ID))
    print('Amount of each class: {}'.format(total_count))

    # # iterate through batches
    # sum_class0 = 0
    # sum_class1 = 0
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print('Batch number: ', i_batch, ', audio: ', sample_batched['audio'].size())
    #     print('loaded_data PHQ Binary Distribution 0/1: {}/{}'.format(len(np.where(sample_batched['phq_binary_gt'].numpy() == 0)[0]), 
    #                                                                   len(np.where(sample_batched['phq_binary_gt'].numpy() == 1)[0])))
    #     print('Participant IDs: {}'.format(sample_batched['ID']))
    #     print('=================================')
    #     sum_class0 += len(np.where(sample_batched['phq_binary_gt'].numpy() == 0)[0])
    #     sum_class1 += len(np.where(sample_batched['phq_binary_gt'].numpy() == 1)[0])

    # print('total chosen amount of data from class 0: {}'.format(sum_class0))
    # print('total chosen amount of data from class 1: {}'.format(sum_class1))

   