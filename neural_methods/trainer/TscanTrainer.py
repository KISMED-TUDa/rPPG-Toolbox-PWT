"""Trainer for TSCAN."""

import datetime
import logging
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image


class TscanTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.train_loss = []
        self.valid_losses = []
        self.visualize_attention_masks = False      # flag whether attention mask visualizations get saved

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("TS-CAN trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 50 == 49:  # print every 50 mini-batches
                    print(f"Train loss: {loss.item()}")
                    print(f'[{epoch}, {idx + 1:5d}] Val loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        print('validation loss: ', np.mean(valid_loss))
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    # visualize attention mask
                    if self.visualize_attention_masks:
                        # show only attention mask of first subject in the middle of movements
                        if _ == 1 and sort_index == 6:
                            path = "C:/Users/Philipp/PycharmProjects/rPPG-Toolbox_Thesis/data/attention_mask/KISMED_v11_rPPG-Benchmark/"

                            # save numpy arrays if you want to plot the signals, using the script: tools/output_signal_viz/plot_gt_and_rppg_bvp_tscan.py
                            # labels_numpy = labels_test.numpy()
                            # pred_ppg_test_numpy = pred_ppg_test.numpy()
                            # np.save(path + 'labels_numpy.npy', labels_numpy)
                            # np.save(path + 'pred_ppg_test_numpy.npy', pred_ppg_test_numpy)

                            # visualize input data
                            for i, input in enumerate(self.model.module.input_data):
                               if i>=2:
                                   self.visualize_attention_mask(input, i-2, path, "input")

                            # Visualize attention masks
                            for i, attention_mask in enumerate(self.model.module.attention_masks):
                               if i>=2:
                                   self.visualize_attention_mask(attention_mask, i-2, path, "attention_mask")

                            # Visualize gated attention masks
                            for i, gated_attention_mask in enumerate(self.model.module.gated_attention_masks):
                                if i>=2:
                                    self.visualize_attention_mask(gated_attention_mask, i-2, path, "gated_attention_mask")


                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def visualize_attention_mask(self, attention_mask, idx_am, path, layer_name):
        font_size = 18
        # set the font to Charter
        font = {'family': 'serif', 'serif': ['Charter'], 'size': font_size}
        plt.rc('font', **font)
        # plt.rc('xtick', labelsize=font_size)
        # plt.rc('ytick', labelsize=font_size)

        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


        if not os.path.exists(path):
            os.makedirs(path)

        # Normalize the attention mask for better visualization
        # attention_mask /= attention_mask.sum(axis=-1, keepdims=True)

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # Frames per second
        video_filename = f"{layer_name}_{idx_am}_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.avi"

        if layer_name != "input":
            video_writer = cv2.VideoWriter(path + video_filename, fourcc, fps, (1807, 1807)) # (attention_mask.shape[2], attention_mask.shape[3]))
        else:
            video_writer = cv2.VideoWriter(path + video_filename, fourcc, fps,
                                           (256, 256))

        # loop through the tensor and save each image
        for i in range(attention_mask.shape[0]):
            if layer_name != "input":
                image = attention_mask[i, 0, :, :]  # Extract the image with shape (70, 70)

                plt.figure(figsize=(5, 5))
                # Use seaborn's heatmap for visualization
                ax = sns.heatmap(image, cmap="viridis", cbar=False, square=True, vmin=attention_mask.min(), vmax=attention_mask.max())   # , annot=False, fmt=".2f")
                ax.tick_params(left=False, bottom=False)
                ax.set(xticklabels=[])
                ax.set(yticklabels=[])

                plt.tight_layout()

                # save the image with an increasing file number
                filename = f"{layer_name}_{idx_am}_{str(i+1).zfill(4)}_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.png"
                plt.savefig(path + filename, dpi=400, bbox_inches='tight', pad_inches=0)

                # Read the saved image using OpenCV
                img = cv2.imread(path + filename)
                # Write the image to the video file
                video_writer.write(img)

                plt.clf()
            else:
                image = attention_mask[i, :, :, :]
                # transpose the dimensions to change the order to (72, 72, 3)
                image_transposed = np.transpose(image, (1, 2, 0))

                # identify indices where the original array is 0
                zero_indices = image_transposed == 0

                # map the non-zero values to the range [0, 255]
                mapped_array = np.interp(image_transposed[~zero_indices], (image_transposed.min(), image_transposed.max()), (0, 255)).astype(np.uint8)

                # set the previous zero values to 0 in the newly mapped array
                mapped_array_with_zeros = np.zeros_like(image_transposed, dtype=np.uint8)
                mapped_array_with_zeros[~zero_indices] = mapped_array

                resized = cv2.resize(mapped_array_with_zeros, (256, 256), interpolation=cv2.INTER_AREA)
                video_writer.write(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        # Release the VideoWriter object
        video_writer.release()
