import torch
import torchvision.datasets as tvds
from paths import datasets_path


class EMNIST(tvds.EMNIST):
    def __init__(self, train=True, class_range=[0, 47], device='cpu', download=True):
        super().__init__(datasets_path, train=train, split='balanced', download=download)

        self.data = self.data.unsqueeze(1).float().div(255).transpose(-1, -2).to(device)
        self.targets = self.targets.to(device)

        idxs = []
        for c in range(class_range[0], class_range[1]):
            idxs.append(torch.where(self.targets==c)[0])
        idxs = torch.cat(idxs)

        self.num_classes = class_range[1] - class_range[0]
        self.data = self.data[idxs]
        self.targets = self.targets[idxs]

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]

        # Ensure image is a float tensor in [0, 1]
        img = img.float() / 255.0 if img.dtype == torch.uint8 else img

        # Shift to [-1, 1]
        img = img * 2.0 - 1.0

        return img, label
