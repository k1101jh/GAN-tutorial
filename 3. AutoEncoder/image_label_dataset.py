import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class ImageLabelDataset(Dataset):
    def __init__(self, image_data_path, csv_path, transform, label=None):
        super(ImageLabelDataset, self).__init__()
        self.data = datasets.ImageFolder(image_data_path,
                                         transform=transform)
        self.num_data = len(self.data)
        self.labels = []
        
        file = open(csv_path, 'r')
        csv_reader = csv.reader(file)
        
        attributes = []
        
        if label is None:
            line = next(csv_reader)
            line = next(csv_reader)
            self.labels = line[1:]
                        
            for line in csv_reader:
                attributes.append([int(x) for x in line[1:]])
        else:
            self.labels.append(label)
            idx = -1
            line = next(csv_reader)
            line = next(csv_reader)
            for i, label_name in enumerate(line, 1):
                if label_name == label:
                    idx = i
            
            assert(idx != -1)
            for _, line in enumerate(csv_reader, 2):
                attributes.append(int(line[idx]))
                        
        self.attribute_data = torch.Tensor(np.array(attributes))
        print(f'Found {self.num_data} images.')
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        image, _ = self.data.__getitem__(idx)
        attribute_val = self.attribute_data[idx]
        
        return image, attribute_val
    
    def get_labels(self):
        return self.labels
    
