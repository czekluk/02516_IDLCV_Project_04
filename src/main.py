from models.early_fusion import EarlyFusionAlexNet
from models.two_stream import TwoStreamAlexNet
from models.aggregate import AggregateAlexNet
from models.cnn_3d import Basic3DCNN, Pretrained3dCNN, PretrainedVisionTransformer
from data_loader.datasets import FrameVideoDataset, FlowVideoDataset
from trainer import FrameVideoTrainer
from torch.utils.data import DataLoader
import os
import torch
import json
from torchvision import transforms
from experiments.experiment import Experiment
from visualizer import Visualizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score
from utils import confusion_matrix_models, plot_confusion_matrix_flow, plot_curves, new_class_weights
import torch.nn as nn

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'ufc10')
DATA_DIR = os.path.join("/dtu/datasets1/02516","ucf101_noleakage")

def early_fusion():
    batch_size = 64
    transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.RandomErasing(p=0.7, scale=(0.15, 0.33), ratio=(0.3, 3.3), value=0)])
    transform_test_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # transform_train = transform_test_val
    
    train_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='train', transform=transform_train, stack_frames = True)
    train_loader = DataLoader(train_set_video,  batch_size=batch_size, shuffle=True)

    val_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='val', transform=transform_test_val, stack_frames = True)
    val_loader = DataLoader(val_set_video,  batch_size=batch_size, shuffle=True)
    
    # Not used in the training process
    test_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='test', transform=transform_test_val, stack_frames = True)
    test_loader = DataLoader(test_set_video,  batch_size=batch_size, shuffle=False)
    
    # Models, Optimizers, Epochs
    models = [EarlyFusionAlexNet]
    optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3, "weight_decay": 1e-4}}]
    epochs = [100]
    
    experiment = Experiment(models, optimizers, epochs, train_loader, val_loader, test_loader, FrameVideoTrainer, "EarlyFusion")
    experiment.run()

def cnn_3d():
    batch_size = 4
    transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    transform_test_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # transform_train = transform_test_val
    
    train_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='train', transform=transform_train, stack_frames = True)
    train_loader = DataLoader(train_set_video,  batch_size=batch_size, shuffle=True)

    val_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='val', transform=transform_test_val, stack_frames = True)
    val_loader = DataLoader(val_set_video,  batch_size=batch_size, shuffle=True)
    
    # Not used in the training process
    test_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='test', transform=transform_test_val, stack_frames = True)
    test_loader = DataLoader(test_set_video,  batch_size=batch_size, shuffle=False)
    # new_weights= new_class_weights("/zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_04/results/20241202-030036_pretrained_3d_cnn/saved_models/20241202-030036_pretrained_3d_cnn_0.2667_Pretrained3dCNN.pth")
    # Models, Optimizers, Epochs
    models = [Pretrained3dCNN]
    optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3, "weight_decay": 1e-4}}]
    epochs = [30]
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = nn.CrossEntropyLoss(weight=new_weights).to(device)

    # experiment = Experiment(
    #     models=models,
    #     optimizers=optimizers,
    #     epochs=epochs,
    #     trainloader=train_loader,
    #     validation_loader=val_loader,
    #     test_loader=test_loader,
    #     trainer=FrameVideoTrainer,
    #     description="cnn_3d",
    #     criterion=criterion
    # )
    experiment = Experiment(
        models=models,
        optimizers=optimizers,
        epochs=epochs,
        trainloader=train_loader,
        validation_loader=val_loader,
        test_loader=test_loader,
        trainer=FrameVideoTrainer,
        description="cnn_3d"
    )
    experiment.run()

def two_stream():
    batch_size = 32
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    transform_test_val = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    # transform_train = transform_test_val
    
    train_set_video = FlowVideoDataset(root_dir=DATA_DIR, split='train', transform=transform_train)
    train_loader = DataLoader(train_set_video,  batch_size=batch_size, shuffle=True)

    val_set_video = FlowVideoDataset(root_dir=DATA_DIR, split='val', transform=transform_test_val)
    val_loader = DataLoader(val_set_video,  batch_size=batch_size, shuffle=True)
    
    # Not used in the training process
    test_set_video = FlowVideoDataset(root_dir=DATA_DIR, split='test', transform=transform_test_val)
    test_loader = DataLoader(test_set_video,  batch_size=batch_size, shuffle=False)
    
    # Models, Optimizers, Epochs
    models = [TwoStreamAlexNet]
    optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3, "weight_decay": 1e-3}}]
    epochs = [50]
    
    experiment = Experiment(models, optimizers, epochs, train_loader, val_loader, test_loader, FrameVideoTrainer, "TwoStream")
    experiment.run()

def aggregate():
    batch_size = 64
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    transform_test_val = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    # transform_train = transform_test_val
    
    train_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='train', transform=transform_train, stack_frames = True)
    train_loader = DataLoader(train_set_video,  batch_size=batch_size, shuffle=True)

    val_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='val', transform=transform_test_val, stack_frames = True)
    val_loader = DataLoader(val_set_video,  batch_size=batch_size, shuffle=True)
    
    # Not used in the training process
    test_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='test', transform=transform_test_val, stack_frames = True)
    test_loader = DataLoader(test_set_video,  batch_size=batch_size, shuffle=False)
    
    # Models, Optimizers, Epochs
    models = [AggregateAlexNet]
    optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3, "weight_decay": 1e-4}}]
    epochs = [20]
    
    experiment = Experiment(models, optimizers, epochs, train_loader, val_loader, test_loader, FrameVideoTrainer, "Aggregate")
    experiment.run()

if __name__ == "__main__":
    # early_fusion()
    # two_stream()
    # plot_confusion_matrix("results/20241129-234934_TwoStream/saved_models/20241129-234934_TwoStream_0.3833_TwoStreamAlexNet.pth")
    #RESULTS_DIR = os.path.join(PROJECT_BASE_DIR, 'results', "20241130-163803_Aggregate" )
    #results_json = "20241130-163803_Aggregate.json"
    #plot_curves(RESULTS_DIR, results_json)
    #aggregate()
    # cnn_3d()
    _,_,_, _ = confusion_matrix_models("/zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_04/results/20241202-152329_pretrained_resnet/saved_models/20241202-152329_cnn_3d_0.1000_Pretrained3dCNN.pth")
    pass
