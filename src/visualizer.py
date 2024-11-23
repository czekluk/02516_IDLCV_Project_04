import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from metrics import calc_accuracy, calc_confusion_matrix

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Visualizer:
    def __init__(self):
        pass

    def plot_loss_accuracy(self, train_loss: list, val_loss: list, train_accuracy: list, val_accuracy: list,
                           save_path: str = os.path.join(PROJECT_BASE_DIR, 'results', 'images'),
                           file_name: str = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_loss_accuracy.png'):
        """
        Plot the loss and accuracy curves. Loss & Accuracy needs to be calculated beforehand.

        Args:
            train_loss (list): Training loss
            val_loss (list): Validation loss
            train_accuracy (list): Training accuracy
            val_accuracy (list): Validation accuracy
            save_path (str, optional): Path to save the plot. Defaults to 'results/images'.
            file_name (str, optional): File name of the plot. Defaults to f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_loss_accuracy.png'.
        """
        # plot the loss and accuracy curves
        fig, axs = plt.subplots(2, 1,figsize=(10, 10))

        axs[0].plot(train_loss, label='Train Loss', color='blue')
        axs[0].plot(val_loss, label='Validation Loss', color='red')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss Curves')
        axs[0].legend()

        axs[1].plot(train_accuracy, label='Train Accuracy', color='blue')
        axs[1].plot(val_accuracy, label='Validation Accuracy', color='red')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy Curves')
        axs[1].legend()

        # save the plots
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, file_name))

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                              save_path: str = os.path.join(PROJECT_BASE_DIR, 'results', 'images'),
                              file_name: str = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_confusion_matrix.png'):
        """
        Plot the confusion matrix. Includes calculation of confusion matrix.

        Args:
            y_true (np.ndarray): True labels of shape (n_videos,)
            y_pred (np.ndarray): Predicted labels of shape (n_videos,)
            num_classes (int): Number of classes
            save_path (str, optional): Path to save the plot. Defaults to 'results/images'.
            file_name (str, optional): File name of the plot. Defaults to f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_confusion_matrix.png'.
        """
        # calculate the confusion matrix
        confusion_matrix = calc_confusion_matrix(y_true, y_pred)

        # plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        x_labels = ['BodyWeightSquats', 'HandstandPushups', 'HandstandWalking', 'JumpingJack', 'JumpRope', 'Lunges', 'PullUps', 'PushUps', 'TrampolineJumping', 'WallPushups']
        y_labels = x_labels
        plt.figure(figsize=(3, 3))        
        sns.heatmap(
            ax=plt.gca(),
            data=confusion_matrix,
            annot=True,
            linewidths=0.5,
            cmap="Reds",
            cbar=False,
            fmt='g',
            xticklabels=x_labels,
            yticklabels=y_labels,
            annot_kws={"size": 18}
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # save the plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, file_name))

    def plot_video_sequence(self, data: np.ndarray, label: np.ndarray, predicted_label: np.ndarray,
                            save_path: str = os.path.join(PROJECT_BASE_DIR, 'results', 'images'),
                            file_name: str = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_video_sequence.png',
                            title: str = 'Video Sequence'):
        """
        Plot a video sequence with predicted & ground truth labels

        Args:
            data (np.ndarray): Video sequence of shape (n_videos, n_frames, channels, height, width)
            label (np.ndarray): Ground truth labels of (n_videos,)
            predicted_label (np.ndarray): Predicted labels of (n_videos,)
            save_path (str, optional): Path to save the plot. Defaults to 'results/images'.
            file_name (str, optional): File name of the plot. Defaults to f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_video_sequence.png'.
            title (str, optional): Title of the plot. Defaults to 'Video Sequence'.
        """
        # metadata for plotting
        n_videos = data.shape[0]
        n_frames = min(data.shape[1], 5)

        # sample frame_idx to plot & sort them in ascending order
        if n_frames < data.shape[1]:
            frame_idx = list(np.random.choice(data.shape[1], n_frames, replace=False))
            frame_idx.sort()
        else:
            frame_idx = list(range(data.shape[1]))
        
        data_labels = ['BodyWeightSquats', 'HandstandPushups', 'HandstandWalking', 'JumpingJack', 'JumpRope', 'Lunges', 'PullUps', 'PushUps', 'TrampolineJumping', 'WallPushups']

        # plot the video sequence
        fig, axs = plt.subplots(n_videos, n_frames, figsize=(n_frames*2, n_videos*2))
        fig.suptitle(title)
        for i in range(n_videos):
            for j in frame_idx:
                axs[i, j].imshow(data[i, j].transpose(1, 2, 0))
                axs[i, j].set_title(f'Label: {data_labels[label[i]]}, Predicted: {data_labels[predicted_label[i]]}')
                axs[i, j].axis('off')

        plt.tight_layout()

        # save the plot to the specified path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, file_name))