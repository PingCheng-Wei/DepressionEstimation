import os
import sys
import math
import time
import shutil
import argparse

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


def main(dataloaders, visual_net, audio_net, fusion_net, evaluator, base_logger, writer, config, args, model_type, ckpt_path):

    if not config['CRITERION']['USE_SOFT_LABEL']:
        assert config['EVALUATOR']['CLASSES_RESOLUTION'] == config['EVALUATOR']['N_CLASSES'], \
            "Argument --config['EVALUATOR']['CLASSES_RESOLUTION'] should be the same as --config['EVALUATOR']['N_CLASSES'] when soft label is not used!"

    model_parameters = [*fusion_net.parameters()] + [*evaluator.parameters()]  # [*visual_net.parameters()] + [*audio_net.parameters()] + [*text_net.parameters()]
    optimizer, scheduler = get_optimizer_scheduler(model_parameters, config['OPTIMIZER'], config['SCHEDULER'])

    test_best_f1_score = 0
    test_epoch_best_f1 = 0
    test_best_acc = 0
    test_epoch_best_acc = 0
    for epoch in range(config['EPOCHS']):
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best Test for f1-score: {test_best_f1_score} at epoch {test_epoch_best_f1}')
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best Test for accuracy: {test_best_acc} at epoch {test_epoch_best_acc}')

        for mode in ['train', 'test']:
            mode_start_time = time.time()

            phq_score_gt = []
            phq_subscores_gt = []
            phq_binary_gt = []
            phq_score_pred = []
            phq_subscores_pred = []
            phq_binary_pred = []

            if mode == 'train':
                visual_net.train()
                audio_net.train()
                # text_net.train()
                fusion_net.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                visual_net.eval()
                audio_net.eval()
                # text_net.eval()
                fusion_net.eval()
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

                # TODO: extract features with multi-model ...
                # combine all models into a function
                def model_processing(input):
                    # get facial visual feature with Deep Visual Net'
                    # input shape for visual_net must be (B, C, F, T) = (batch_size, channels, features, time series)
                    B, T, F, C = input['visual'].shape
                    visual_input = input['visual'].permute(0, 3, 2, 1).contiguous()
                    visual_features = visual_net(visual_input.to(args.device))  # output dim: [B, visual net output dim]

                    # get audio feature with Deep Audio Net'
                    # input shape for audio_net must be (B, F, T) = (batch_size, features, time series)
                    B, F, T = input['audio'].shape
                    audio_input = input['audio'].view(B, F, T)
                    audio_features = audio_net(audio_input.to(args.device))  # output dim: [B, audio net output dim]

                    # # get Text features with Deep Text Net'
                    # # input shape for text_net must be (B, F, T) = (batch_size, features, time series))
                    # B, T, F = input['text'].shape
                    # text_input = input['text'].permute(0, 2, 1).contiguous()
                    # text_features = text_net(text_input.to(args.device))  # output dim: [B, text net output dim]

                    # -------------------------------------- features fusion --------------------------------------
                    # combine all features into shape: B, C=1, num_modal, audio net output dim
                    all_features = torch.stack([visual_features, audio_features], dim=1).unsqueeze(dim=1)
                    fused_features = fusion_net(all_features)
                    B, C, H, W = fused_features.shape
                    fused_features = fused_features.view(B, -1)  # shape: (B, num_modal x audio net output dim)

                    # start evaluating and get probabilities
                    """ Start evaluating ...
                    Arguments:
                        'features' should have size [batch_size, input_feature_dim]
                    Output:
                        if PREDICT_TYPE == phq-subscores:
                            'probs' is a list of torch matrices
                            len(probs) == number of subscores == 8
                            probs[0].size() == (batch size, class resolution)
                        else:
                            'probs' a torch matrices with shape: (batch size, class resolution)
                        
                    """
                    probs = evaluator(fused_features)

                    return probs
                
                if mode == 'train':
                    
                    # choose the right GT for criterion based on prediciton type
                    gt = get_gt(data, config['EVALUATOR']['PREDICT_TYPE'])

                    # get dynamic weights for cross entropy loss if needed
                    if config['CRITERION']['USE_WEIGHTS']:
                        config['CRITERION']['WEIGHTS'] = get_crossentropy_weights(gt, config['EVALUATOR'])
                    criterion = get_criterion(config['CRITERION'], args)

                    if config['OPTIMIZER']['USE_SAM']:
                        models = [visual_net, audio_net, fusion_net, evaluator]
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

            # store the model score
            if mode == 'train':
                train_model_f1_score = f1_score
                train_model_acc = accuracy*100
            elif mode == 'test':
                test_model_f1_score = f1_score
                test_model_acc = accuracy*100

            # tensorboard
            writer.add_scalars(f'Classification/{mode}/TPR_TNR', {'TPR': tpr,
                                                                  'TNR': tnr}, epoch)
            writer.add_scalars(f'Classification/{mode}/Scores', {'Precision': precision,
                                                                 'Recall': recall,
                                                                 'F1_score': f1_score}, epoch)
            writer.add_scalars('Regression/Scores', {'MAE': mae,
                                                     'MSE': mse,
                                                     'RMSE': rmse,
                                                     'R2': r2}, epoch)
            writer.add_scalar('Spearman_correlation/{}'.format(mode), rho, epoch)

        if test_model_f1_score >= test_best_f1_score:
            test_best_f1_score = test_model_f1_score
            test_epoch_best_f1 = epoch
            
            msg = (f'--------- New best found for f1-score at epoch {epoch} !!! ---------\n'
                   f'- train score: {train_model_f1_score:8.6f}\n'
                   f'- test score: {test_model_f1_score:8.6f}\n'
                   f'--------- New best found for f1-score at epoch {epoch} !!! ---------\n')
            log_and_print(base_logger, msg)

            if args.save:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')
                file_path = os.path.join(ckpt_path, '{}_{}_f1_score-{:6.4f}.pt'.format(model_type, timestamp, test_best_f1_score))

                torch.save({'epoch': epoch,
                            'visual_net': visual_net.state_dict(),
                            'audio_net': audio_net.state_dict(),
                            'fusion_net': fusion_net.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_f1_score': test_best_f1_score},
                        file_path)
        
        if test_model_acc >= test_best_acc:
            test_best_acc = test_model_acc
            test_epoch_best_acc = epoch
            
            msg = (f'--------- New best found for accuracy at epoch {epoch} !!! ---------\n'
                   f'- train score: {train_model_acc:8.6f}\n'
                   f'- test score: {test_model_acc:8.6f}\n'
                   f'--------- New best found for accuracy at epoch {epoch} !!! ---------\n')
            log_and_print(base_logger, msg)

            if args.save:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')
                file_path = os.path.join(ckpt_path, '{}_{}_acc-{:6.4f}.pt'.format(model_type, timestamp, test_best_acc))

                torch.save({'epoch': epoch,
                            'visual_net': visual_net.state_dict(),
                            'audio_net': audio_net.state_dict(),
                            'fusion_net': fusion_net.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_acc': test_best_acc},
                        file_path)
        
        # update lr with scheduler
        scheduler.step()
                        
    if args.save:
        # save the best model of all epochs in ['MODEL']['WEIGHTS']['PATH']
        best_model_weights_path = find_last_ckpts(ckpt_path, model_type, date=None)
        shutil.copy(best_model_weights_path, config['WEIGHTS']['PATH'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        help="path to yaml file",
                        required=False,
                        default='config/config_phq-subscores.yaml')
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
    visual_net, audio_net, fusion_net, evaluator = get_models(config['MODEL'], args, model_type, ckpt_path)

    main(dataloaders, visual_net, audio_net, fusion_net, evaluator, base_logger, writer, config['MODEL'], args, model_type, ckpt_path)

    writer.close()
