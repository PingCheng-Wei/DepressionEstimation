'''
DAIC-WOZ Depression Database

The package includes 189 folders of sessions 300-492 and
other documents as well as matlab scripts in util.zip
Excluded sessions: 342,394,398,460
-------------------------------------------------------------------------
run from the commend line as such:
    in Linux: python3 download_DAIC-WOZ.py --out_dir=D:\test
'''

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def get_logger(filepath, log_title):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 54 + log_title + '-' * 54)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


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


def load_keypoints(keypoints_path):
    fkps_df = pre_check(pd.read_csv(keypoints_path, low_memory=False))
    # process into format TxVxC
    fkps_conf = fkps_df[' confidence'].to_numpy()
    x_coor = min_max_scaler(fkps_df[fkps_df.columns[4: 72]].to_numpy())
    y_coor = min_max_scaler(fkps_df[fkps_df.columns[72: 140]].to_numpy())
    z_coor = min_max_scaler(fkps_df[fkps_df.columns[140: 208]].to_numpy())
    fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)
    T, V, C = fkps_coor.shape

#     # initialize the final facial key points which contains coordinate and confidence score
#     fkps_final = np.zeros((T, V, C+1))

#     fkps_final[:, :, :3] = fkps_coor
#     for i in range(V):
#         fkps_final[:, i, 3] = fkps_conf

    return fkps_coor


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


if __name__ == '__main__':

    # initialization
    dataset_rootdir = '/cvhci/temp/wpingcheng/DAIC-WOZ_dataset'
    file_name = 'KP_movement_ranking.log'
    base_logger = get_logger(file_name, '----- the movement difference between each Facial Key Point -----')
    all_sessions = list(range(300, 492 + 1))
    # all_sessions = list(range(300, 305))  # debug
    excluded = [342, 394, 398, 460]
    final_result = np.zeros(68)
    # for the csv file generation
    columns = ['Participent ID'] + ['KP_{:02}'.format(i+1) for i in range(68)]
    data = [] 

    for ID in all_sessions:
        if ID not in excluded:
            print(f'Calculating KP Movement in Participant {ID} ......')
            participant = f"{ID}_P"
            fkps_3D_name = f"{ID}_CLNF_features3D.txt"
            transcipt = f"{ID}_TRANSCRIPT.csv"

            # read the 3D facial keypoints text file by using pd.read_csv
            text_path = os.path.join(dataset_rootdir, participant, transcipt)
            fkps_path = os.path.join(dataset_rootdir, participant, fkps_3D_name)
            text_df = pd.read_csv(text_path, sep='\t').fillna('')

            fkps_coor = load_keypoints(fkps_path)
            fkps_coor = visual_clipping(fkps_coor, 30, text_df)
            fkps_diff = fkps_coor[1:]
            fkps_diff = fkps_diff - fkps_coor[:-1]
            fkps_diff = np.linalg.norm(fkps_diff, axis=-1).sum(axis=0)

            KP_IDs_LowToHigh = np.argsort(fkps_diff) + 1
            reward_points = np.argsort(KP_IDs_LowToHigh)
            final_result += reward_points
            data.append([ID] + list(reward_points))

    ID_ranking_HighToLow = np.argsort(final_result)[::-1] + 1
    log_and_print(base_logger, f'The final ranking of Key Point ID based on movement (high -> low):\n{ID_ranking_HighToLow}')

    df = pd.DataFrame(data, columns=columns)
    df.to_csv('Reward_Points_KP_movement.csv', index=False)
            





