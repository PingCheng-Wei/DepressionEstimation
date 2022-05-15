'''
Generate a new database from DAIC-WOZ Database

 Symbol convention: 
- N: batch size
- T: number of samples per batch (each sample includes _v_ key points)
- V: number of key point per person
- M: number of person 
- C: information per point, like (x,y,z)-Coordinate or (x,y,z,p)-Coordinate + Confident probability  

Architecture of generated database:
[DAIC_WOZ-generated_database]
 └── [train]
      ├── [original_data]
      │    ├── [facial_keypoints]
      │    │    └── only_coordinate, shape of each npy file: (1800, 68, 3)
      │    ├── [gaze_vectors]
      │    │    └── only_coordinate, shape of each npy file: (1800, 4, 3)
      │    ├── [action_units]
      │    │    └── regression values, shape of each npy file: (1800, 14)
      │    ├── [position_rotation]
      │    │    └── only_coordinate, shape of each npy file: (1800, 6)
      │    ├── [hog_features]
      │    │    └── hog feature values, shape of each npy file: (1800, 4464)
      │    ├── [audio]
      │    │    ├── [spectrogram], shape of each npy file: (1025, 1800)
      │    │    └── [Mel-spectrogram], shape of each npy file: (80, 1800)
      │    ├── [text]
      │    │    └── [sentence_embeddings], shape of each npy file: (10, 768)
      │    ├── PHQ_Binary_GT.npy
      │    ├── PHQ_Score_GT.npy
      │    ├── PHQ_Subscore_GT.npy
      │    └── PHQ_Gender_GT.npy
      └── [clipped_data]
           ├── [facial_keypoints]
           │    └── only_coordinate, shape of each npy file: (1800, 68, 3)
           ├── [gaze_vectors]
           │    └── only_coordinate, shape of each npy file: (1800, 4, 3)
           ├── [action_units]
           │    └── regression values, shape of each npy file: (1800, 14)
           ├── [position_rotation]
           │    └── only_coordinate, shape of each npy file: (1800, 6)
           ├── [hog_features]
           │    └── hog feature values, shape of each npy file: (1800, 4464)
           ├── [audio]
           │    ├── [spectrogram], shape of each npy file: (1025, 1800)
           │    └── [Mel-spectrogram], shape of each npy file: (80, 1800)
           ├── [text]
           │    └── [sentence_embeddings], shape of each npy file: (10, 768)
           ├── PHQ_Binary_GT.npy
           ├── PHQ_Score_GT.npy
           ├── PHQ_Subscore_GT.npy
           └── PHQ_Gender_GT.npy
 '''

import os
import struct
import numpy as np
import pandas as pd
import wave
import librosa
import tensorflow_hub as hub


def create_folders(root_dir):
    folders = ['original_data', 'clipped_data']
    subfolders = ['facial_keypoints', 'gaze_vectors', 'action_units', 
                  'position_rotation', 'hog_features', 'audio', 'text']
    audio_subfolders = ['spectrogram', 'mel-spectrogram']

    os.makedirs(root_dir, exist_ok=True)
    for i in folders:
        for j in subfolders:
            if j == 'audio':
                for m in audio_subfolders:
                    os.makedirs(os.path.join(root_dir, i, j, m), exist_ok=True)
            else:
                os.makedirs(os.path.join(root_dir, i, j), exist_ok=True)

                            
def min_max_scaler(data):
    '''recale the data, which is a 2D matrix, to 0-1'''
    return (data - data.min())/(data.max() - data.min())


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def pre_check(data_df):
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    data_np = data_df.to_numpy()
    data_min = data_np[np.where(~(np.isnan(data_np[:, 4:])))].min()
    data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
    return data_df


def load_gaze(gaze_path):
    gaze_df = pre_check(pd.read_csv(gaze_path, low_memory=False))
    # process into format TxVxC
    gaze_coor = gaze_df.iloc[:, 4:].to_numpy().reshape(len(gaze_df), 4, 3)  # 4 gaze vectors, 3 axes
    T, V, C = gaze_coor.shape

    # # initialize the final gaze_3D which contains coordinate and confidence score
    # gaze_final = np.zeros((T, V, C+1))

    # gaze_conf = gaze_df[' confidence'].to_numpy()
    # gaze_final[:, :, :3] = gaze_coor
    # for i in range(V):
    #     gaze_final[:, i, 3] = gaze_conf

    return gaze_coor


def load_keypoints(keypoints_path):
    fkps_df = pre_check(pd.read_csv(keypoints_path, low_memory=False))
    # process into format TxVxC
    x_coor = min_max_scaler(fkps_df[fkps_df.columns[4: 72]].to_numpy())
    y_coor = min_max_scaler(fkps_df[fkps_df.columns[72: 140]].to_numpy())
    z_coor = min_max_scaler(fkps_df[fkps_df.columns[140: 208]].to_numpy())
    fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)
    T, V, C = fkps_coor.shape

    # # initialize the final facial key points which contains coordinate and confidence score
    # fkps_final = np.zeros((T, V, C+1))

    # fkps_final[:, :, :3] = fkps_coor
    # fkps_conf = fkps_df[' confidence'].to_numpy()
    # for i in range(V):
    #     fkps_final[:, i, 3] = fkps_conf

    return fkps_coor


def load_AUs(AUs_path):

    def check_AUs(data_df):
        data_df = data_df.apply(pd.to_numeric, errors='coerce')
        data_np = data_df.to_numpy()
        data_min = data_np[np.where(~(np.isnan(data_np[:, 4:18])))].min()
        data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
        return data_df
    
    AUs_df = check_AUs(pd.read_csv(AUs_path, low_memory=False))
    AUs_features = min_max_scaler(AUs_df.iloc[:, 4:18].to_numpy())

    return AUs_features


def load_pose(pose_path):
    pose_df = pre_check(pd.read_csv(pose_path, low_memory=False))
    pose_coor = pose_df.iloc[:, 4:].to_numpy()
    T, C = pose_coor.shape
    
    # initialize the final pose features which contains coordinate
    pose_features = np.zeros((T, C))
    # normalize the position coordinates part
    norm_part = min_max_scaler(pose_coor[:, :3])

    pose_features[:, :3] = norm_part         # normalized position coordinates
    pose_features[:, :3] = pose_coor[:, :3]  # head rotation coordinates
    pose_features = pose_features.reshape(T, 2, 3) # 2 coordinates, 3 axes

    return pose_features


def read_hog(filename, batch_size=5000):
    """
    Citation: https://gist.github.com/btlorch/6d259bfe6b753a7a88490c0607f07ff8
    Read HoG features file created by OpenFace.
    For each frame, OpenFace extracts 12 * 12 * 31 HoG features, i.e., num_features = 4464. These features are stored in row-major order.
    :param filename: path to .hog file created by OpenFace
    :param batch_size: how many rows to read at a time
    :return: is_valid, hog_features
        is_valid: ndarray of shape [num_frames]
        hog_features: ndarray of shape [num_frames, num_features]
    """
    all_feature_vectors = []
    with open(filename, "rb") as f:
        num_cols, = struct.unpack("i", f.read(4))
        num_rows, = struct.unpack("i", f.read(4))
        num_channels, = struct.unpack("i", f.read(4))

        # The first four bytes encode a boolean value whether the frame is valid
        num_features = 1 + num_rows * num_cols * num_channels
        feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
        feature_vector = np.array(feature_vector).reshape((1, num_features))
        all_feature_vectors.append(feature_vector)

        # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
        num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
        # Read in batches of given batch_size
        num_floats_to_read = num_floats_per_feature_vector * batch_size
        # Multiply by 4 because of float32
        num_bytes_to_read = num_floats_to_read * 4

        while True:
            bytes = f.read(num_bytes_to_read)
            # For comparison how many bytes were actually read
            num_bytes_read = len(bytes)
            assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
            num_floats_read = num_bytes_read // 4
            assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
            num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

            feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
            # Convert to array
            feature_vectors = np.array(feature_vectors).reshape((num_feature_vectors_read, num_floats_per_feature_vector))
            # Discard the first three values in each row (num_cols, num_rows, num_channels)
            feature_vectors = feature_vectors[:, 3:]
            # Append to list of all feature vectors that have been read so far
            all_feature_vectors.append(feature_vectors)

            if num_bytes_read < num_bytes_to_read:
                break

        # Concatenate batches
        all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

        # Split into is-valid and feature vectors
        is_valid = all_feature_vectors[:, 0]
        feature_vectors = all_feature_vectors[:, 1:]

        return is_valid, feature_vectors


def load_hog(hog_path):
    _, hog_features = read_hog(hog_path)  # feature dimension: 4464
    return hog_features


def load_audio(audio_path):
    wavefile = wave.open(audio_path)
    audio_sr = wavefile.getframerate()
    n_samples = wavefile.getnframes()
    signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)
    
    return signal.astype(float), audio_sr
   
    
def visual_clipping(visual_data, visual_sr, text_df):
    counter = 0
    for t in text_df.itertuples():
        if getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            else:
                start = getattr(t, 'start_time')
                stop = getattr(t, 'stop_time')
                start_sample = int(start * visual_sr)
                stop_sample = int(stop * visual_sr)
                if counter == 0:
                    edited_vdata = visual_data[start_sample:stop_sample]
                else:
                    edited_vdata = np.vstack((edited_vdata, visual_data[start_sample:stop_sample]))
                
                counter += 1

    return edited_vdata
    
    
def audio_clipping(audio, audio_sr, text_df, zero_padding=False):
    if zero_padding:
        edited_audio = np.zeros(audio.shape[0])
        for t in text_df.itertuples():
            if getattr(t, 'speaker') == 'Participant':
                if 'scrubbed_entry' in getattr(t,'value'):
                    continue
                else:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * audio_sr)
                    stop_sample = int(stop * audio_sr)
                    edited_audio[start_sample:stop_sample] = audio[start_sample:stop_sample]
        
        # cut head and tail of interview
        first_start = text_df['start_time'][0]
        last_stop = text_df['stop_time'][len(text_df)-1]
        edited_audio = edited_audio[int(first_start*audio_sr):int(last_stop*audio_sr)]
    
    else:
        edited_audio = []
        for t in text_df.itertuples():
            if getattr(t,'speaker') == 'Participant':
                if 'scrubbed_entry' in getattr(t,'value'):
                    continue
                else:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * audio_sr)
                    stop_sample = int(stop * audio_sr)
                    edited_audio = np.hstack((edited_audio, audio[start_sample:stop_sample]))

    return edited_audio
   
    
def convert_spectrogram(audio, frame_size=2048, hop_size=533):
    # extracting with Short-Time Fourier Transform
    S_scale = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)
    spectrogram = np.abs(S_scale) ** 2
    # convert amplitude to DBs
    log_spectrogram = librosa.power_to_db(spectrogram)
    
    return log_spectrogram  # in dB


def convert_mel_spectrogram(audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80):
    mel_spectrogram = librosa.feature.melspectrogram(audio, 
                                                     sr=audio_sr, 
                                                     n_fft=frame_size, 
                                                     hop_length=hop_size,
                                                     n_mels=num_mel_bands)
    # convert amplitude to DBs
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    return log_mel_spectrogram  # in dB


def sentence_embedding(text_df, model):
    sentences = []
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            else:
                sentences.append(getattr(t, 'value'))
    
    return model.encode(sentences)


def use_embedding(text_df, model):
    sentences = []
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            else:
                sentences.append(getattr(t, 'value'))
    
    return model(sentences).numpy()  # output size: (sentences length, 512)


def get_num_frame(data, frame_size, hop_size):
    T = data.shape[0]
    if (T - frame_size) % hop_size == 0:
        num_frame = (T - frame_size) // hop_size + 1
    else:
        num_frame = (T - frame_size) // hop_size + 2
    return num_frame


def get_text_hop_size(text, frame_size, num_frame):
    T = text.shape[0]
    return (T - frame_size) // (num_frame - 1)
    
    
def visual_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple()
        size = size + (pad_size,) + data.shape[1:]
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data
    
    return padded_data


def audio_padding(data, pad_size):
    if data.shape[1] != pad_size:
        size = tuple((data.shape[0], pad_size))
        padded_data = np.zeros(size)
        padded_data[:, :data.shape[1]] = data
    else:
        padded_data = data
    
    return padded_data


def text_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple((pad_size, data.shape[1]))
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data
    
    return padded_data


def random_shift_fkps(fkps_coor, fkps_coor_conf):
    shifted_fc = np.copy(fkps_coor)
    shifted_fcc = np.copy(fkps_coor_conf)
    
    for i in range(3):
        factor = np.random.uniform(-0.05, 0.05)
        shifted_fc[:, :, i] = shifted_fc[:, :, i] + factor
        shifted_fcc[:, :, i] = shifted_fcc[:, :, i] + factor
    
    return shifted_fc, shifted_fcc

    
def sliding_window(fkps_features, gaze_features, AUs_features, pose_features, 
                   hog_features, spectro, mel_spectro, text_feature, visual_sr, 
                   window_size, overlap_size, output_root, ID):

    
    frame_size = window_size * visual_sr
    hop_size = (window_size - overlap_size) * visual_sr
    num_frame = get_num_frame(fkps_features, frame_size, hop_size)
    text_frame_size = 10
    text_hop_size = get_text_hop_size(text_feature, text_frame_size, num_frame)
    
    if ID > 485:
        print('creating the data from the rest of the participants')
        # start sliding through and generating data
        for i in range(num_frame):
            frame_sample_fkps = visual_padding(fkps_features[i*hop_size:i*hop_size+frame_size], frame_size)
            frame_sample_gaze = visual_padding(gaze_features[i*hop_size:i*hop_size+frame_size], frame_size)
            frame_sample_AUs = visual_padding(AUs_features[i*hop_size:i*hop_size+frame_size], frame_size)
            frame_sample_pose = visual_padding(pose_features[i*hop_size:i*hop_size+frame_size], frame_size)
            frame_sample_hog = visual_padding(hog_features[i*hop_size:i*hop_size+frame_size], frame_size)
            frame_sample_spec = audio_padding(spectro[:, i*hop_size:i*hop_size+frame_size], frame_size)
            frame_sample_mspec = audio_padding(mel_spectro[:, i*hop_size:i*hop_size+frame_size], frame_size)
            frame_sample_text = text_padding(text_feature[i*text_hop_size:i*text_hop_size+text_frame_size], text_frame_size)
            
            # start storing
            np.save(os.path.join(output_root, 'facial_keypoints', f'{ID}-{i:02}_kps.npy'), frame_sample_fkps)
            np.save(os.path.join(output_root, 'gaze_vectors', f'{ID}-{i:02}_gaze.npy'), frame_sample_gaze)
            np.save(os.path.join(output_root, 'action_units', f'{ID}-{i:02}_AUs.npy'), frame_sample_AUs)
            np.save(os.path.join(output_root, 'position_rotation', f'{ID}-{i:02}_pose.npy'), frame_sample_pose)
            np.save(os.path.join(output_root, 'hog_features', f'{ID}-{i:02}_hog.npy'), frame_sample_hog)
            np.save(os.path.join(output_root, 'audio', 'spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_spec)
            np.save(os.path.join(output_root, 'audio', 'mel-spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_mspec)
            np.save(os.path.join(output_root, 'text', f'{ID}-{i:02}_text.npy'), frame_sample_text)
    else:
        print('pass')

    return num_frame




if __name__ == '__main__':

    # output root
    root = '/cvhci/temp/wpingcheng'
    root_dir = os.path.join(root, 'DAIC_WOZ-generated_database_V2', 'train')
    create_folders(root_dir)
    np.random.seed(1)

    # read training gt file
    gt_path = '/cvhci/data/depression/DAIC-WOZ_dataset/full_train_split_Depression_AVEC2017.csv'
    gt_df = pd.read_csv(gt_path) 

    # initialization
    use_embed_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    window_size = 60   # 60s
    overlap_size = 10  # 10s
    GT = {'original_data': {'ID_gt':[], 'gender_gt': [], 'phq_binary_gt': [], 'phq_score_gt':[], 'phq_subscores_gt':[]}, 
          'clipped_data': {'ID_gt':[], 'gender_gt': [], 'phq_binary_gt': [], 'phq_score_gt':[], 'phq_subscores_gt':[]}}
    
    for i in range(len(gt_df)):
        # extract training gt details
        patient_ID = gt_df['Participant_ID'][i]
        phq_binary_gt = gt_df['PHQ8_Binary'][i]
        phq_score_gt = gt_df['PHQ8_Score'][i]
        gender_gt = gt_df['Gender'][i]
        phq_subscores_gt = gt_df.iloc[i, 4:].to_numpy().tolist()
        print(f'Processing Participant {patient_ID}, Gender: {gender_gt} ...')
        print(f'- PHQ Binary: {phq_binary_gt}, PHQ Score: {phq_score_gt}, Subscore: {phq_subscores_gt}')

        # get all files path of participant
        keypoints_path = f'/cvhci/data/depression/DAIC-WOZ_dataset/{patient_ID}_P/{patient_ID}_CLNF_features3D.txt'
        gaze_path = f'/cvhci/data/depression/DAIC-WOZ_dataset/{patient_ID}_P/{patient_ID}_CLNF_gaze.txt'
        AUs_path = f'/cvhci/data/depression/DAIC-WOZ_dataset/{patient_ID}_P/{patient_ID}_CLNF_AUs.txt'
        pose_path = f'/cvhci/data/depression/DAIC-WOZ_dataset/{patient_ID}_P/{patient_ID}_CLNF_pose.txt'
        hog_path = f'/cvhci/data/depression/DAIC-WOZ_dataset/{patient_ID}_P/{patient_ID}_CLNF_hog.bin'
        audio_path = f'/cvhci/data/depression/DAIC-WOZ_dataset/{patient_ID}_P/{patient_ID}_AUDIO.wav'
        text_path = f'/cvhci/data/depression/DAIC-WOZ_dataset/{patient_ID}_P/{patient_ID}_TRANSCRIPT.csv'

        # read transcipt file
        text_df = pd.read_csv(text_path, sep='\t').fillna('')
        first_start_time = text_df['start_time'][0]
        last_stop_time = text_df['stop_time'][len(text_df)-1]
        
        # read & process visual files
        gaze_features = load_gaze(gaze_path)
        fkps_features = load_keypoints(keypoints_path)
        AUs_features = load_AUs(AUs_path)
        pose_features = load_pose(pose_path)
        hog_features = load_hog(hog_path)
        visual_sr = 30  # 30Hz

        # read audio file
        audio, audio_sr = load_audio(audio_path)

        # extract text feature
        text_feature = use_embedding(text_df, model=use_embed_large)


        ########################################
        # feature extraction for original_data #
        ########################################

        print(f'Extracting feature of Participant {patient_ID} for original_data...')

        # visual
        filtered_fkps_features = fkps_features[int(first_start_time*visual_sr):int(last_stop_time*visual_sr)]
        filtered_gaze_features = gaze_features[int(first_start_time*visual_sr):int(last_stop_time*visual_sr)]
        filtered_AUs_features = AUs_features[int(first_start_time*visual_sr):int(last_stop_time*visual_sr)]
        filtered_pose_features = pose_features[int(first_start_time*visual_sr):int(last_stop_time*visual_sr)]
        filtered_hog_features = hog_features[int(first_start_time*visual_sr):int(last_stop_time*visual_sr)]

        # audio
        filtered_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=True)
        # spectrogram, mel spectrogram
        spectro = normalize(convert_spectrogram(filtered_audio, frame_size=2048, hop_size=533))
        mel_spectro = normalize(convert_mel_spectrogram(filtered_audio, audio_sr, 
                                                        frame_size=2048, hop_size=533, num_mel_bands=80))

        ###################################################################
        # start creating data in 'original_data' folder #
        ###################################################################

        output_root = os.path.join(root_dir, 'original_data')
        num_frame = sliding_window(filtered_fkps_features, filtered_gaze_features, filtered_AUs_features, 
                                   filtered_pose_features, filtered_hog_features, spectro, mel_spectro, 
                                   text_feature, visual_sr, window_size, overlap_size, output_root, patient_ID)

        # replicate GT
        for _ in range(num_frame):
            GT['original_data']['ID_gt'].append(patient_ID)
            GT['original_data']['gender_gt'].append(gender_gt)
            GT['original_data']['phq_binary_gt'].append(phq_binary_gt)
            GT['original_data']['phq_score_gt'].append(phq_score_gt)
            GT['original_data']['phq_subscores_gt'].append(phq_subscores_gt)

        #######################################
        # feature extraction for clipped_data #
        #######################################

        print(f'Extracting feature of Participant {patient_ID} for clipped_data...')

        # visual
        clipped_fkps_features = visual_clipping(fkps_features, visual_sr, text_df)
        clipped_gaze_features = visual_clipping(gaze_features, visual_sr, text_df)
        clipped_AUs_features = visual_clipping(AUs_features, visual_sr, text_df)
        clipped_pose_features = visual_clipping(pose_features, visual_sr, text_df)
        clipped_hog_features = visual_clipping(hog_features, visual_sr, text_df)

        # audio
        clipped_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=False)
        # spectrogram, mel spectrogram
        spectro = normalize(convert_spectrogram(clipped_audio, frame_size=2048, hop_size=533))
        mel_spectro = normalize(convert_mel_spectrogram(clipped_audio, audio_sr, 
                                                        frame_size=2048, hop_size=533, num_mel_bands=80))

        ##################################################################
        # start creating data in 'clipped_data' folder #
        ##################################################################

        output_root = os.path.join(root_dir, 'clipped_data')
        num_frame = sliding_window(clipped_fkps_features, clipped_gaze_features, clipped_AUs_features, 
                                   clipped_pose_features, clipped_hog_features, spectro, mel_spectro, 
                                   text_feature, visual_sr, window_size, overlap_size, output_root, patient_ID)

        # replicate GT
        for _ in range(num_frame):
            GT['clipped_data']['ID_gt'].append(patient_ID)
            GT['clipped_data']['gender_gt'].append(gender_gt)
            GT['clipped_data']['phq_binary_gt'].append(phq_binary_gt)
            GT['clipped_data']['phq_score_gt'].append(phq_score_gt)
            GT['clipped_data']['phq_subscores_gt'].append(phq_subscores_gt)

        print(f'Participant {patient_ID} done!')
        print('='*50)

    # store new GT
    for k1, v1 in GT.items():
        for k2, v2 in v1.items():
            # print(os.path.join(root_dir, k1, k2, f'{k3}.npy'))
            np.save(os.path.join(root_dir, k1, f'{k2}.npy'), v2)

    print('All done!')


