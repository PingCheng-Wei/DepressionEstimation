import os
import time
import shutil
import argparse
from unicodedata import decimal
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime
from scipy import stats
from autolab_core import YamlConfig

# local functions
from utils import *
from models.bypass_bn import enable_running_stats, disable_running_stats


def main(dataloaders, audio_net, evaluator, base_logger, writer, config, args, model_type, ckpt_path):

    if not config['CRITERION']['USE_SOFT_LABEL']:
        assert config['EVALUATOR']['CLASSES_RESOLUTION'] == config['EVALUATOR']['N_CLASSES'], \
            "Argument --config['EVALUATOR']['CLASSES_RESOLUTION'] should be the same as --config['EVALUATOR']['N_CLASSES'] when soft label is not used!"

    model_parameters = [*audio_net.parameters()] + [*evaluator.parameters()]
    optimizer, scheduler = get_optimizer_scheduler(model_parameters, config['OPTIMIZER'], config['SCHEDULER'])

    test_best_f1_score = 0
    test_epoch_best_f1 = 0
    test_best_acc = 0
    test_epoch_best_acc = 0
    for epoch in range(config['EPOCHS']):
        
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best Test for f1-score: {test_best_f1_score} at epoch {test_epoch_best_f1}')
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best Test for accuracy: {test_best_acc} at epoch {test_epoch_best_acc}')

        for mode in ['test']:
            mode_start_time = time.time()

            phq_score_gt = []
            phq_subscores_gt = []
            phq_binary_gt = []
            phq_score_pred = []
            phq_subscores_pred = []
            phq_binary_pred = []
            gender_gt = []
            ID_gt = []

            if mode == 'train':
                audio_net.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                audio_net.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            total_loss = 0
            log_interval_loss = 0
            log_interval = 10
            batch_number = 0
            n_batches = len(dataloaders[mode])
            batches_start_time = time.time()
            
            for data in tqdm(dataloaders[mode]):
                batch_size = data['ID'].size(0)

                # store ground truth
                phq_score_gt.extend(data['phq_score_gt'].numpy().astype(float))  # 1D list
                phq_binary_gt.extend(data['phq_binary_gt'].numpy().astype(float))  # 1D list
                # store gender and ID GT
                gender_gt.extend(data['gender_gt'].numpy().astype(float))  # 1D list
                ID_gt.extend(data['ID'].numpy().astype(float))  # 1D list

                # TODO: extract features with multi-model ...
                # combine all models into a function
                def model_processing(input):
                    # get audio feature with Deep Audio Net'
                    # input shape for audio_net must be (B, H, W) = (batch_size, freq(height), width)
                    B, H, W = input['audio'].shape
                    audio_input = input['audio'].view(B, H, W)
                    audio_features = audio_net(audio_input.to(args.device))  # output dim: [B, audio net output dim]

                    # start evaluating and get probabilities
                    """ Start evaluating ...
                    Arguments:
                        'text_features' should have size [batch_size, input_feature_dim]
                    Output:
                        if PREDICT_TYPE == phq-subscores:
                            'probs' is a list of torch matrices
                            len(probs) == number of subscores == 8
                            probs[0].size() == (batch size, class resolution)
                        else:
                            'probs' a torch matrices with shape: (batch size, class resolution)
                        
                    """
                    probs = evaluator(audio_features)

                    return probs
                
                if mode == 'train':
                    
                    # choose the right GT for criterion based on prediciton type
                    gt = get_gt(data, config['EVALUATOR']['PREDICT_TYPE'])

                    # get dynamic weights for cross entropy loss if needed
                    if config['CRITERION']['USE_WEIGHTS']:
                        config['CRITERION']['WEIGHTS'] = get_crossentropy_weights(gt, config['EVALUATOR'])
                    criterion = get_criterion(config['CRITERION'], args)

                    if config['OPTIMIZER']['USE_SAM']:
                        models = [audio_net, evaluator]
                        # first forward-backward pass
                        for model in models:
                            enable_running_stats(model)      # to avoid potentially problems if you use batch normalization
                        probs = model_processing(input=data)  # start prediction 
                        loss = compute_loss(criterion, probs, gt, config['EVALUATOR'], args, 
                                            use_soft_label=config['CRITERION']['USE_SOFT_LABEL'])
                        loss.backward()                      # backpropagation - use this loss for any training statistics
                        optimizer.first_step(zero_grad=True)

                        # second forward-backward pass
                        for model in models:
                            disable_running_stats(model)
                        compute_loss(criterion, model_processing(input=data), gt, config['EVALUATOR'], 
                                    args, use_soft_label=config['CRITERION']['USE_SOFT_LABEL']).backward()
                        optimizer.second_step(zero_grad=True)
                    
                    else:
                        # only one time forward-backward pass 
                        probs = model_processing(input=data)
                        loss = compute_loss(criterion, probs, gt, config['EVALUATOR'], args, 
                                            use_soft_label=config['CRITERION']['USE_SOFT_LABEL'])
                        optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(models_parameters, max_norm=2.0, norm_type=2)
                        optimizer.step()

                else:
                    # for test set, only do prediction
                    probs = model_processing(input=data)

                # predict the final score
                pred_score = compute_score(probs, config['EVALUATOR'], args)
                phq_score_pred.extend([pred_score[i].item() for i in range(batch_size)])  # 1D list
                phq_binary_pred.extend([1 if pred_score[i].item() >= config['PHQ_THRESHOLD'] else 0 for i in range(batch_size)])
                # phq_binary_pred.extend([pred_score[i].item() for i in range(batch_size)])
                
                if mode == 'train':
                    # information per batch
                    total_loss += loss.item()
                    log_interval_loss += loss.item()
                    if batch_number % log_interval == 0 and batch_number > 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - batches_start_time) * 1000 / log_interval
                        current_loss = log_interval_loss / log_interval
                        print(f'| epoch {epoch:3d} | {mode} | {batch_number:3d}/{n_batches:3d} batches | '
                            f'LR {lr:7.6f} | ms/batch {ms_per_batch:5.2f} | loss {current_loss:8.5f} |')

                        # tensorboard
                        writer.add_scalar('Loss_per_{}_batches/{}'.format(log_interval, mode),
                                        current_loss, epoch*n_batches+batch_number)

                        log_interval_loss = 0
                        batches_start_time = time.time()
                else:
                    # for test set we don't need to calculate the loss so just leave it 'nan'
                    total_loss = np.nan

                batch_number += 1

            # information per mode
            print('PHQ Score prediction: {}'.format(phq_score_pred[:20]))
            print('PHQ Score ground truth: {}'.format(phq_score_gt[:20]))

            print('PHQ Binary prediction: {}'.format(phq_binary_pred[:20]))
            print('PHQ Binary ground truth: {}'.format(phq_binary_gt[:20]))

            average_loss = total_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            s_per_mode = time.time() - mode_start_time
            accuracy, correct_number = get_accuracy(phq_binary_gt, phq_binary_pred)

            # store information in logger and print
            print('-' * 110)
            msg = ('  End of {0}:\n  | time: {1:8.3f}s | LR: {2:7.6f} | Average Loss: {3:8.5f} | Accuracy: {4:5.2f}%'
                   ' ({5}/{6}) |').format(mode, s_per_mode, lr, average_loss, accuracy*100, correct_number, len(phq_binary_gt))
            log_and_print(base_logger, msg)
            print('-' * 110)

            # tensorboard
            writer.add_scalar('Loss_per_epoch/{}'.format(mode), average_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(mode), accuracy*100, epoch)
            writer.add_scalar('Learning_rate/{}'.format(mode), lr, epoch)

            # Calculating additional evaluation scores
            log_and_print(base_logger, '  Output Scores:')

            # confusion matrix
            [[tp, fp], [fn, tn]] = standard_confusion_matrix(phq_binary_gt, phq_binary_pred)
            msg = (f'  - Confusion Matrix:\n'
                '    -----------------------\n'
                f'    | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                '    -----------------------\n'
                f'    | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                '    -----------------------')
            log_and_print(base_logger, msg)

            # classification related
            tpr, tnr, precision, recall, f1_score = get_classification_scores(phq_binary_gt, phq_binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)

            # regression related
            mae, mse, rmse, r2 = get_regression_scores(phq_score_gt, phq_score_pred)
            msg = ('  - Regression:\n'
                   '      MAE: {0:7.4f}\n'
                   '      MSE: {1:7.4f}\n'
                   '      RMSE: {2:7.4f}\n'
                   '      R2: {3:7.4f}\n').format(mae, mse, rmse, r2)
            log_and_print(base_logger, msg)

            # Calculate a Spearman correlation coefficien
            rho, p = stats.spearmanr(phq_score_gt, phq_score_pred)  # phq_binary_gt, phq_binary_pred
            msg = ('  - Correlation:\n'
                   '      Spearman correlation: {0:8.6f}\n').format(rho)
            log_and_print(base_logger, msg)

            # -------------------------------------- Gender Analysis --------------------------------------
            log_and_print(base_logger, 'Gender Analysis based on clips:')
            gender_gt = np.asarray(gender_gt)
            phq_binary_gt = np.asarray(phq_binary_gt)
            phq_binary_pred = np.asarray(phq_binary_pred)
            phq_score_pred = np.asarray(phq_score_pred)

            # get female data (Gender 0)
            female_idx = np.where(gender_gt==0)[0]
            female_phq_binary_gt = phq_binary_gt[female_idx]
            female_phq_binary_pred = phq_binary_pred[female_idx]
            # GT
            num_female_total = len(female_phq_binary_gt)
            num_female_depressed = len(np.where(female_phq_binary_gt==1)[0])
            num_female_notdepressed = len(np.where(female_phq_binary_gt==0)[0])
            log_and_print(base_logger, f'  Female - Number of clips: {num_female_total}, depressed: {num_female_depressed}, non-depressed: {num_female_notdepressed}')
            # score & confusion matrix
            accuracy, correct_number = get_accuracy(female_phq_binary_gt, female_phq_binary_pred)
            [[tp, fp], [fn, tn]] = standard_confusion_matrix(female_phq_binary_gt, female_phq_binary_pred)
            msg = (f'    - Confusion Matrix ( acc: {accuracy*100:5.2f}% ({correct_number}/{len(female_phq_binary_gt)}) ):\n'
                '      -----------------------\n'
                f'      | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                '      -----------------------\n'
                f'      | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                '      -----------------------')
            log_and_print(base_logger, msg)
            # classification related
            tpr, tnr, precision, recall, f1_score = get_classification_scores(female_phq_binary_gt, female_phq_binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)

            # get male data (Gender 1)
            male_idx = np.where(gender_gt==1)[0]
            male_phq_binary_gt = phq_binary_gt[male_idx]
            male_phq_binary_pred = phq_binary_pred[male_idx]
            # GT
            num_male_total = len(male_phq_binary_gt)
            num_male_depressed = len(np.where(male_phq_binary_gt==1)[0])
            num_male_notdepressed = len(np.where(male_phq_binary_gt==0)[0])
            log_and_print(base_logger, f'  Male - Number of clips: {num_male_total}, depressed: {num_male_depressed}, non-depressed: {num_male_notdepressed}')
            # score & confusion matrix
            accuracy, correct_number = get_accuracy(male_phq_binary_gt, male_phq_binary_pred)
            [[tp, fp], [fn, tn]] = standard_confusion_matrix(male_phq_binary_gt, male_phq_binary_pred)
            msg = (f'    - Confusion Matrix ( acc: {accuracy*100:5.2f}% ({correct_number}/{len(male_phq_binary_gt)}) ):\n'
                '      -----------------------\n'
                f'      | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                '      -----------------------\n'
                f'      | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                '      -----------------------\n')
            log_and_print(base_logger, msg)
            # classification related
            tpr, tnr, precision, recall, f1_score = get_classification_scores(male_phq_binary_gt, male_phq_binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)


            # -------------------------------------- Participant Analysis --------------------------------------
            log_and_print(base_logger, 'Participant Analysis:')
            ID_gt = np.asarray(ID_gt)
            IDs, IDs_count = np.unique(ID_gt, return_counts=True)
            # for the csv file generation
            columns = columns = ['Participent ID', 'Gender', 'Binary GT', 'Final binary prediction', 'num_clips'] + ['clip_{:02}'.format(i+1) for i in range(IDs_count.max())]
            data = [] 
            threshold = 0.5
            final_gender_gt = []
            final_phq_binary_gt = []
            final_phq_binary_pred = []
            for i, id in enumerate(IDs):
                id_indices = np.where(ID_gt==id)[0]
                id_clips_gt = phq_binary_gt[id_indices]
                id_gender_gt = gender_gt[id_indices]
                id_clips_pred = np.zeros(IDs_count.max())
                id_clips_pred[:] = None 
                id_clips_pred[:IDs_count[i]] = phq_binary_pred[id_indices]
                id_clips_phq_pred = np.zeros(IDs_count.max())
                id_clips_phq_pred[:] = None 
                id_clips_phq_pred[:IDs_count[i]] = phq_score_pred[id_indices]
                # integer = int(IDs_count[i] * threshold)
                # decimal = IDs_count[i] * threshold - int(IDs_count[i] * threshold)
                final_id_pred = 1 if id_clips_pred[:IDs_count[i]].sum() >= (IDs_count[i] * threshold) else 0
                data.append([id, id_gender_gt[0], id_clips_gt[0], final_id_pred, IDs_count[i]] + list(id_clips_pred))
                data.append([id, id_gender_gt[0], id_clips_gt[0], final_id_pred, IDs_count[i]] + list(id_clips_phq_pred))
                final_gender_gt.append(int(id_gender_gt[0]))
                final_phq_binary_gt.append(int(id_clips_gt[0]))
                final_phq_binary_pred.append(int(final_id_pred))

            log_and_print(base_logger, f'   Binary GT of Participants: {final_phq_binary_gt}')
            log_and_print(base_logger, f'  Prediction of Participants: {final_phq_binary_pred}')
            accuracy, correct_number = get_accuracy(final_phq_binary_gt, final_phq_binary_pred)
            [[tp, fp], [fn, tn]] = standard_confusion_matrix(final_phq_binary_gt, final_phq_binary_pred)
            msg = (f'    - Confusion Matrix ( acc: {accuracy*100:5.2f}% ({correct_number}/{len(final_phq_binary_gt)}) ):\n'
                '      -----------------------\n'
                f'      | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                '      -----------------------\n'
                f'      | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                '      -----------------------\n')
            log_and_print(base_logger, msg)
            # classification related
            tpr, tnr, precision, recall, f1_score = get_classification_scores(final_phq_binary_gt, final_phq_binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)

            df = pd.DataFrame(data, columns=columns)
            df.fillna('')
            df.to_csv('Participant Analysis.csv', index=False)

            # -------------------------------------- Participant Gender Analysis --------------------------------------
            log_and_print(base_logger, 'Participant gender Analysis:')
            final_gender_gt = np.asarray(final_gender_gt)
            final_phq_binary_gt = np.asarray(final_phq_binary_gt)
            final_phq_binary_pred = np.asarray(final_phq_binary_pred)

            # get female data (Gender 0)
            female_idx = np.where(final_gender_gt==0)[0]
            female_phq_binary_gt = final_phq_binary_gt[female_idx]
            female_phq_binary_pred = final_phq_binary_pred[female_idx]
            # GT
            num_female_total = len(female_phq_binary_gt)
            num_female_depressed = len(np.where(female_phq_binary_gt==1)[0])
            num_female_notdepressed = len(np.where(female_phq_binary_gt==0)[0])
            log_and_print(base_logger, f'  Female - Number of participants: {num_female_total}, depressed: {num_female_depressed}, non-depressed: {num_female_notdepressed}')
            # score & confusion matrix
            accuracy, correct_number = get_accuracy(female_phq_binary_gt, female_phq_binary_pred)
            [[tp, fp], [fn, tn]] = standard_confusion_matrix(female_phq_binary_gt, female_phq_binary_pred)
            msg = (f'    - Confusion Matrix ( acc: {accuracy*100:5.2f}% ({correct_number}/{len(female_phq_binary_gt)}) ):\n'
                '      -----------------------\n'
                f'      | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                '      -----------------------\n'
                f'      | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                '      -----------------------\n')
            log_and_print(base_logger, msg)
            # classification related
            tpr, tnr, precision, recall, f1_score = get_classification_scores(female_phq_binary_gt, female_phq_binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)

            # get male data (Gender 1)
            male_idx = np.where(final_gender_gt==1)[0]
            male_phq_binary_gt = final_phq_binary_gt[male_idx]
            male_phq_binary_pred = final_phq_binary_pred[male_idx]
            # GT
            num_male_total = len(male_phq_binary_gt)
            num_male_depressed = len(np.where(male_phq_binary_gt==1)[0])
            num_male_notdepressed = len(np.where(male_phq_binary_gt==0)[0])
            log_and_print(base_logger, f'  Male - Number of participants: {num_male_total}, depressed: {num_male_depressed}, non-depressed: {num_male_notdepressed}')
            # score & confusion matrix
            accuracy, correct_number = get_accuracy(male_phq_binary_gt, male_phq_binary_pred)
            [[tp, fp], [fn, tn]] = standard_confusion_matrix(male_phq_binary_gt, male_phq_binary_pred)
            msg = (f'    - Confusion Matrix ( acc: {accuracy*100:5.2f}% ({correct_number}/{len(male_phq_binary_gt)}) ):\n'
                '      -----------------------\n'
                f'      | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                '      -----------------------\n'
                f'      | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                '      -----------------------\n')
            log_and_print(base_logger, msg)
                        # classification related
            tpr, tnr, precision, recall, f1_score = get_classification_scores(male_phq_binary_gt, male_phq_binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        help="path to yaml file",
                        required=False,
                        default='config/config_inference.yaml')
    parser.add_argument('--device',
                        type=str,
                        help="set up torch device: 'cpu' or 'cuda' (GPU)",
                        required=False,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # remember to set the gpu device number
    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        required=False,
                        default='2, 3')
    parser.add_argument('--save',
                        type=bool,
                        help='if set true, save the best model',
                        required=False,
                        default=False)
    args = parser.parse_args()

    # set up GPU
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load config file into dict() format
    config = YamlConfig(args.config_file)

    # create the output folder (name of experiment) for storing model result such as logger information
    if not os.path.exists(config['OUTPUT_DIR']):
        os.mkdir(config['OUTPUT_DIR'])
    # create the root folder for storing checkpoints during training
    if not os.path.exists(config['CKPTS_DIR']):
        os.mkdir(config['CKPTS_DIR'])
    # create the subfolder for storing checkpoints based on the model type
    if not os.path.exists(os.path.join(config['CKPTS_DIR'], config['TYPE'])):
        os.mkdir(os.path.join(config['CKPTS_DIR'], config['TYPE']))
    # create the folder for storing the best model after all epochs
    if not os.path.exists(config['MODEL']['WEIGHTS']['PATH']):
        os.mkdir(config['MODEL']['WEIGHTS']['PATH'])

    # print configuration
    print('=' * 40)
    print(config.file_contents)
    config.save(os.path.join(config['OUTPUT_DIR'], config['SAVE_CONFIG_NAME']))
    print('=' * 40)

    # initialize random seed for torch and numpy
    init_seed(config['MANUAL_SEED'])

    # get logger os.path.join(config['OUTPUT_DIR'], f'{config['TYPE']}_{config['LOG_TITLE']}.log')
    file_name = os.path.join(config['OUTPUT_DIR'], '{}.log'.format(config['TYPE']))
    base_logger = get_logger(file_name, config['LOG_TITLE'])
    # get summary writer for TensorBoard
    writer = SummaryWriter(os.path.join(config['OUTPUT_DIR'], 'runs'))
    # get dataloaders
    dataloaders = get_dataloaders(config['DATA'])
    # get models
    ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
    model_type = config['TYPE']
    audio_net, evaluator = get_models(config['MODEL'], args, model_type, ckpt_path)

    main(dataloaders, audio_net, evaluator, base_logger, writer, config['MODEL'], args, model_type, ckpt_path)

    writer.close()
