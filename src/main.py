from models.early_fusion import EarlyFusionAlexNet
from models.two_stream import TwoStreamAlexNet
from data_loader.datasets import FrameVideoDataset, FlowVideoDataset
from trainer import FrameVideoTrainer
from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
from experiments.experiment import Experiment


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
    epochs = [5]
    
    experiment = Experiment(models, optimizers, epochs, train_loader, val_loader, test_loader, FrameVideoTrainer, "TwoStream")
    experiment.run()

if __name__ == "__main__":
    #early_fusion()
    two_stream()
