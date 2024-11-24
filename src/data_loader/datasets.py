from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data/ufc10')

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
    stack_frames = True
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
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        """Get metadata for a given attribute and value. Used for reading labels."""
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=NUM_CLASSES)

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
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


if __name__ == '__main__':
    # Specify params for dataset
    print('Data Directory:', DATA_DIR)
    root_dir = DATA_DIR
    transform = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor()])
    
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
