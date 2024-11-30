from models.early_fusion import EarlyFusionAlexNet
from models.two_stream import TwoStreamAlexNet
from models.aggregate import AggregateAlexNet
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

def plot_confusion_matrix_flow(model_path: str, results_dir: str = "results/"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_test_val = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    model = torch.load(model_path, weights_only=False)
    model.eval()
    test_set_video = FlowVideoDataset(root_dir=DATA_DIR, split='test', transform=transform_test_val)
    test_loader = DataLoader(test_set_video,  batch_size=1, shuffle=False)
    
    y_true_list = []
    y_pred_list = []

    test_correct = 0
    for minibatch_no, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        data, target = data.to(device), target.to(device)  # [batch, channels, frames, height, width]
        with torch.no_grad():
            output = model(data)  # [batch_size, n_classes]

        # Predicted classes
        predicted_classes = torch.argmax(output, dim=1)  # Shape: [batch_size]
        target_classes = torch.argmax(target, dim=1)     # Shape: [batch_size]

        # Collect predictions and true labels
        y_pred_list.append(predicted_classes.cpu().numpy())  # Convert to numpy
        y_true_list.append(target_classes.cpu().numpy())     # Convert to numpy

        # Accuracy tracking (optional)
        test_correct += (target_classes == predicted_classes).sum().cpu().item()

    # Combine all batches into final arrays
    y_true = np.concatenate(y_true_list, axis=0)  # Shape: [total_samples]
    y_pred = np.concatenate(y_pred_list, axis=0)  # Shape: [total_samples]
    
    visualizer = Visualizer()
    visualizer.plot_confusion_matrix(y_true, y_pred, 10, save_path=results_dir)
    
def plot_curves(results_dir, results_json):
    with open(os.path.join(results_dir, results_json), 'r') as file:
        results = json.load(file)

    vis = Visualizer()
    vis.plot_loss_accuracy(train_loss=results[0]["train_loss"], val_loss=results[0]["val_loss"],
                           train_accuracy=results[0]["train_acc"], val_accuracy=results[0]["val_acc"],
                           save_path=results_dir)

if __name__ == "__main__":
    # early_fusion()
    # two_stream()
    # plot_confusion_matrix("results/20241129-234934_TwoStream/saved_models/20241129-234934_TwoStream_0.3833_TwoStreamAlexNet.pth")
    #RESULTS_DIR = os.path.join(PROJECT_BASE_DIR, 'results', "20241130-163803_Aggregate" )
    #results_json = "20241130-163803_Aggregate.json"
    #plot_curves(RESULTS_DIR, results_json)
    #aggregate()
    pass
