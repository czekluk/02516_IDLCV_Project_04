from glob import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.functional import F

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'ufc10')
DATA_DIR = os.path.join("/dtu/datasets1/02516","ucf101_noleakage")

NUM_CLASSES = 10

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir=DATA_DIR,
    split='train',
    transform=None
    ):
        """
        Dataset for frame-by-frame loading of UFC-10 dataset. Used for training per-frame models.
        
        Returns data in the format [channels, height, width].

        Args:
            root_dir (str, optional): Root directory of data folder 'ufc10'. Defaults to DATA_DIR specified in file.
            split (str, optional): Whether to return 'train' or 'test' split. Defaults to 'train'.
            transform (PyTorch transform, optional): Transform to apply to dataset. Defaults to None.
        """
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        """Get metadata for a given attribute and value. Used for reading labels."""
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=NUM_CLASSES)
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = transforms.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = DATA_DIR, 
    split = 'train', 
    transform = None,
    stack_frames = True,
    sampled_frames = 10
        ):
        """
        Dataset for VIDEO loading of UFC-10 dataset. 
        Used for training early fusion/late fusion and 3D CNNs and validation of per-frame models.
        
        Returns data in the format [channels, frames, height, width].
        
        Args:
            root_dir (str, optional): Root directory of data folder 'ufc10'. Defaults to DATA_DIR specified in file.
            split (str, optional): Whether to return 'train' or 'test' split. Defaults to 'train'.
            transform (PyTorch transform, optional): Transform to apply to dataset. Defaults to None.
            stack_frames (bool, optional): Whether to stack frames into a single tensor [C, T, H, W]. Defaults to True.
        """
        # Use os.path.join to construct platform-independent paths
        videos_path = os.path.join(root_dir, "videos", split, "*", "*.avi")
        self.video_paths = sorted(glob(videos_path))
        metadata_path = os.path.join(root_dir, "metadata", f"{split}.csv")
        self.df = pd.read_csv(metadata_path)

        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = sampled_frames

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        """Get metadata for a given attribute and value. Used for reading labels."""
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract video name without extension

        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=NUM_CLASSES)

        # Construct the path to the frames directory
        video_frames_dir = video_path.replace(os.path.join("videos", ""), os.path.join("frames", "")).rsplit(".avi", 1)[0]
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [transforms.ToTensor()(frame) for frame in video_frames]
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames
    
class FlowVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = DATA_DIR, 
    split = 'train', 
    transform = None
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform   
        self.n_sampled_frames = 1

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=NUM_CLASSES)

        video_flows_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'flows')
        flows = self.load_flows(video_flows_dir)
        video_frames_dir = video_path.replace(os.path.join("videos", ""), os.path.join("frames", "")).rsplit(".avi", 1)[0]
        frames = self.load_frames(video_frames_dir)
        # pick one random frame
        #frame = frames[np.random.choice(len(frames))]
        frame = frames[0]
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = transforms.ToTensor()(frame)

        # stack flows and frame together - frame should be first three channels
        final_frame = torch.concat([frame, flows], dim=0)

        return final_frame, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

    def load_flows(self, flows_dir):
        flows = []
        for i in range(1, self.n_sampled_frames + 1):
            flow_file = os.path.join(f'{flows_dir}', f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            if self.transform:
                flow = self.transform(flow)
            else:
                flow = transforms.ToTensor()(flow)
            flows.append(flow)
        flows = torch.stack(flows)

        return flows.flatten(0,1)


if __name__ == '__main__':
    # Specify params for dataset
    print('Data Directory:', DATA_DIR)
    root_dir = DATA_DIR
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((64, 64))])
    
    # Test Frame Dataset
    print('-'*50)
    print('Testing FrameImageDataset:')
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    frameimage_loader = DataLoader(frameimage_dataset, batch_size=8, shuffle=False)
    for frames, labels in frameimage_loader:
        print("Data shape [batch, channels, height, width]:", frames.shape)
        print("Labels shape [batch, n_classes]:", labels.shape)
        break
    
    # Test Video Dataset - Stack Frames into Tensor
    print('-'*50)
    print('Testing FrameVideoDataset: (tensor)')
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    for video_frames, labels in framevideostack_loader:
        print("Data shape [batch, channels, frames, height, width]:", video_frames.shape)
        print("Labels shape [batch, n_classes]:", labels.shape)
        break

    # Test Video Dataset - Do not Stack Frames
    print('-'*50)
    print('Testing FrameVideoDataset: (list)')
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
    framevideolist_loader = DataLoader(framevideolist_dataset, batch_size=8, shuffle=False)
    for video_frames, labels in framevideolist_loader:
        print("Data length [frames], list:", len(video_frames))
        for frame in video_frames:
            print("Data shape [batch, channels, height, width]:", frame.shape)
        print("Labels shape [batch, n_classes]:", labels.shape)
        break

    # Test Flow Video Dataset
    print('-'*50)
    print('Testing FlowVideoDataset:')
    flowvideo_dataset = FlowVideoDataset(root_dir=root_dir, split="val", transform=transform)
    flowvideo_loader = DataLoader(flowvideo_dataset, batch_size=8, shuffle=False)
    for frames, labels in flowvideo_loader:
        print("Data shape [batch, channels, frames, height, width]:", frames.shape)
        print("Labels shape [batch, n_classes]:", labels.shape)
        break
