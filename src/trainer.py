import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from torchvision import transforms
from models.basic_models import DummyCNN
from data_loader.datasets import FrameImageDataset, FrameVideoDataset
from models.late_fusion import LateFusionAlexNet
from models.aggregate import AggregateDummyNet, AggregateAlexNet

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'ufc10')

class FrameVideoTrainer:
    
    def __init__(self, models: List[nn.Module], optimizer_functions: List[dict], 
                 epochs: int, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, description: str="", criterion= nn.CrossEntropyLoss()) -> None:
        """
        Class for training different models with different optimizers and different numbers of epochs.
        
        Args:   models              -   list of models. The models are not instances but classes. example: [AlexNet, ResNet]
                optimizer_funcitons -   list of dictionaries specifying different optimizers.
                                        example: optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}]
                epochs              -   list of different epochs to train. example: [10, 15]
                train_loader        -   torch.utils.data.DataLoader
                test_loader         -   torch.utils.data.DataLoader
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion
        self.models = models
        self.optimizer_functions = optimizer_functions
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.description = description
    
    
    def train(self) -> List[dict]:
        """
        Train the different models, with different optimizers, with different number of epochs.
        
        Returns:    List of dictionaries representing different experiments.
                    The list is sorted in descending order based on the achieved accuracy
                    after the final epoch.
        """
        outputs = []
        for network in self.models:
            for optimizer_config in self.optimizer_functions:
                for epoch_no in self.epochs:
                    print("#########################################################")
                    print(f"Training model: {network.__name__}")
                    print(f"Optimizer: {optimizer_config['optimizer'].__name__}")
                    print(f"Training for {epoch_no} epochs")
                    model = network()
                    out_dict = self._train_single_configuration(model, optimizer_config, epoch_no)
                    outputs.append(out_dict)
        outputs_sorted = sorted(outputs, key=lambda x: x['val_acc'][-1], reverse=True)
        return outputs_sorted
    
    
    def _train_single_configuration(self, model: nn.Module, optimizer_config: dict, num_epochs: int) -> dict:
        model.to(self.device)
        optimizer = optimizer_config["optimizer"](model.parameters(), **optimizer_config["params"])
        
        out_dict = {
            'model_name':       model.__class__.__name__,
            'model':            model,
            'train_acc':        [],
            'val_acc':         [],
            'train_loss':       [],
            'val_loss':        [],
            'test_acc':         0,
            'epochs':           num_epochs,
            'optimizer_config': optimizer_config, 
            'description':      self.description
            }
        
        for epoch in tqdm(range(num_epochs), unit='epoch'):
            model.train()
            train_correct = 0
            train_loss = []
            
            for minibatch_no, (data, targets) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                data, targets = data.to(self.device), targets.to(self.device) # [batch=num_videos, channels, num_frames_per_video, height, width]

                optimizer.zero_grad()
                output = model(data) # [batch_size, n_classes]
                predicted_classes = torch.argmax(output, dim=1)
                target_classes = torch.argmax(targets, dim=1)

                loss = self.criterion(output, targets.clone().detach().float().requires_grad_(True)) # correct shapes for nn.CrossEntropyLoss
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
                train_correct += (target_classes==predicted_classes).sum().cpu().item()
                
            val_loss = []
            val_correct = 0
            model.eval()
            for minibatch_no, (data, target) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                data, target = data.to(self.device), target.to(self.device)  # [batch, channels, frames, height, width]
                with torch.no_grad():
                    output = model(data)  # [batch_size, n_classes]

                # Calculate the loss
                predicted_classes = torch.argmax(output, dim=1)
                target_classes = torch.argmax(target, dim=1)
                loss = self.criterion(output, target_classes)

                val_loss.append(loss.cpu().item()) 
                val_correct += (target_classes==predicted_classes).sum().cpu().item()
            
            out_dict['train_acc'].append(train_correct/len(self.train_loader.dataset))
            out_dict['val_acc'].append(val_correct/len(self.val_loader.dataset))
            out_dict['train_loss'].append(np.mean(train_loss))
            out_dict['val_loss'].append(np.mean(val_loss))
            print(f"Loss train: {np.mean(train_loss):.3f}\t Val: {np.mean(val_loss):.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t Val: {out_dict['val_acc'][-1]*100:.1f}%")
        print(f"Loss train: {np.mean(train_loss):.3f}\t Val: {np.mean(val_loss):.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t Val: {out_dict['val_acc'][-1]*100:.1f}%")
        
        test_correct = 0
        for minibatch_no, (data, target) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            data, target = data.to(self.device), target.to(self.device)  # [batch, channels, frames, height, width]
            with torch.no_grad():
                output = model(data)  # [batch_size, n_classes]

            # Calculate the loss
            predicted_classes = torch.argmax(output, dim=1)
            target_classes = torch.argmax(target, dim=1)
            loss = self.criterion(output, target_classes)

            test_correct += (target_classes==predicted_classes).sum().cpu().item()
        
        correct_perc = test_correct/len(self.test_loader.dataset)
        out_dict['test_acc'] = correct_perc
        print(f"Test Acc: {correct_perc * 100:.1f}%")    
        
        
        return out_dict

def test_latefusion_trainer():
    # Testing Late Fusion Trainer:
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    train_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='train', transform=transform, stack_frames = True)
    train_loader = DataLoader(train_set_video,  batch_size=2, shuffle=True)

    val_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='val', transform=transform, stack_frames = True)
    val_loader = DataLoader(val_set_video,  batch_size=2, shuffle=True)

    # Not used in the training process
    test_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='test', transform=transform, stack_frames = True)
    test_loader = DataLoader(test_set_video,  batch_size=2, shuffle=False)
    
    trainer = FrameVideoTrainer(
        models = [LateFusionAlexNet],
        optimizer_functions = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}],
        epochs = [2],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    outputs = trainer.train()
    return outputs
    
def test_aggregate_trainer():
    # Testing Late Fusion Trainer:
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    train_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='train', transform=transform, stack_frames = True)
    train_loader = DataLoader(train_set_video,  batch_size=2, shuffle=True)

    val_set_video = FrameVideoDataset(split='val', transform=transform, stack_frames = True)
    val_loader = DataLoader(val_set_video,  batch_size=2, shuffle=True)

    # Not used in the training process
    test_set_video = FrameVideoDataset(split='test', transform=transform, stack_frames = True)
    test_loader = DataLoader(test_set_video,  batch_size=2, shuffle=False)
    
    trainer = FrameVideoTrainer(
        models = [AggregateAlexNet],
        optimizer_functions = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}],
        epochs = [2],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    outputs = trainer.train()
    return outputs

if __name__ == "__main__":
    output = test_aggregate_trainer()
    print(output)
