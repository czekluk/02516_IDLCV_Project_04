from data_loader.datasets import FrameVideoDataset, FlowVideoDataset
from torch.utils.data import DataLoader
import os
import torch
import json
from torchvision import transforms
from visualizer import Visualizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'ufc10')
DATA_DIR = os.path.join("/dtu/datasets1/02516","ucf101_noleakage")

def confusion_matrix_models(model_path: str, results_dir: str = "results/"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=1
    transform_test_val = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    model = torch.load(model_path, weights_only=False)
    model.eval()
    test_set_video = FrameVideoDataset(root_dir=DATA_DIR, split='test', transform=transform_test_val, stack_frames = True)
    test_loader = DataLoader(test_set_video,  batch_size=batch_size, shuffle=True)
    
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
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, average='weighted', labels=np.arange(10))
    recall = recall_score(y_true, y_pred, average='weighted', labels=np.arange(10))

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    # Calculate accuracy for all classes
    class_accuracies = {}
    for class_idx in range(10):
        class_mask = (y_true == class_idx)
        class_correct = (y_pred[class_mask] == y_true[class_mask]).sum()
        class_total = class_mask.sum()
        class_accuracies[class_idx] = class_correct / class_total if class_total > 0 else 0.0

    for class_idx, accuracy in class_accuracies.items():
        print(f"Accuracy for class {class_idx}: {accuracy:.4f}")

    # Calculate class prevalence
    class_prevalence = _calculate_class_prevalence(y_true)
    print("Class Prevalence:", class_prevalence)

    return class_accuracies, precision, recall, class_prevalence

def _calculate_class_prevalence(y_true):
    class_prevalence = {}
    total_samples = len(y_true)
    for class_idx in range(10):
        class_prevalence[class_idx] = np.sum(y_true == class_idx) / total_samples
    return class_prevalence

def _suggest_weights(class_accuracies, class_prevalence):
    weights = {}
    for class_idx in range(10):
        accuracy = class_accuracies.get(class_idx, 0.0)
        prevalence = class_prevalence.get(class_idx, 0.0)
        if prevalence > 0:
            weights[class_idx] = (1.0 - accuracy) / prevalence
        else:
            weights[class_idx] = 0.0

        # Normalize weights to sum up to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for class_idx in weights:
                weights[class_idx] /= total_weight
        else:
            print("Warning: All classes have prevalence 0. Setting all weights to 1.")
            for class_idx in weights:
                weights[class_idx] = 1.0 / len(weights)
    return weights

def new_class_weights(model_path: str):
    acc, _, _, prev= confusion_matrix_models(model_path)
    weights = _suggest_weights(acc, prev)
    weights_tensor = torch.tensor([weights[class_idx] for class_idx in range(10)], dtype=torch.float32)
    return weights_tensor


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