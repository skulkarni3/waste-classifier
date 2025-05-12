import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image
import numpy as np

# Category mapping for the household waste dataset
CATEGORY_MAPPING_HOUSEHOLD_WASTE = {
    'recyclable': [
        'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
        'cardboard_boxes', 'cardboard_packaging', 'glass_beverage_bottles',
        'glass_cosmetic_containers', 'glass_food_jars', 'magazines',
        'newspaper', 'office_paper', 'plastic_detergent_bottles',
        'plastic_food_containers', 'plastic_soda_bottles',
        'plastic_water_bottles', 'steel_food_cans','plastic_cup_lids'
    ],
    'compostable': [
        'coffee_grounds', 'eggshells', 'food_waste', 'tea_bags', 'paper_cups'
    ],
    'landfill': [ 'disposable_plastic_cutlery','plastic_shopping_bags', 'plastic_straws',
        'plastic_trash_bags',  'styrofoam_cups',
        'styrofoam_food_containers', 'shoes', 'clothing'
    ]
}

# Category mapping for the garbage dataset
CATEGORY_MAPPING_GARBAGE_DATASET = {
    'recyclable': [
        'metal', 'glass', 'paper', 'cardboard', 'plastic'
    ],
    'compostable': [
        'biological'
    ],
    'landfill': [
        'trash', 'shoes', 'clothes'
    ]
}
# Category mapping for the real waste dataset
CATEGORY_MAPPING_REALWASTE = {
    'recyclable': [
        'Metal', 'Glass', 'Paper', 'Cardboard', 'Plastic'
    ],
    'compostable': [
        'Food Organics', 'Vegetation'
    ],
    'landfill': [
        'Miscellaneous Trash', 'Textile Trash'
    ]
}


class WasteClassifierCNN(nn.Module):
    def __init__(self):
        super(WasteClassifierCNN, self).__init__()
        
        # Convolutional layers with max pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 3)  # Output 3 classes: Recyclable, Compostable, Landfill
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Increased dropout for better regularization
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 512 * 8 * 8)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def compute_class_weights(dataset):
    """Compute class weights based on class frequencies."""
    class_counts = {0: 0, 1: 0, 2: 0}
    for _, label in dataset:
        class_counts[label] += 1
    
    total_samples = sum(class_counts.values())
    class_weights = {
        class_idx: total_samples / (len(class_counts) * count)
        for class_idx, count in class_counts.items()
    }
    
    # Convert to tensor
    weights = torch.tensor([class_weights[i] for i in range(len(class_counts))])
    return weights

class CombinedWasteDataset(Dataset):
    def __init__(self, waste_classification_path, household_waste_path, garbage_dataset_path, realwaste_path, transform=None, train=True, use_only_landfill_from_new=False):
        self.transform = transform
        self.data = []
        
        # Load waste classification dataset (only TRAIN or TEST based on train parameter)
        split = 'TRAIN' if train else 'TEST'
        base_path = os.path.join(waste_classification_path, split)

        # Load Recyclable (R) class
        r_path = os.path.join(base_path, 'R')
        if os.path.isdir(r_path):
            for img_name in os.listdir(r_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append((os.path.join(r_path, img_name), 0))  # Recyclable = 0
               
        # Load Organic/Compostable (O) class
        o_path = os.path.join(base_path, 'O')
        if os.path.isdir(o_path):
            for img_name in os.listdir(o_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append((os.path.join(o_path, img_name), 1))  # Compostable = 1
                  
       

        # For household waste dataset, we'll split it based on the train parameter
        additional_waste_data = []
        for category in os.listdir(household_waste_path):
            if category in CATEGORY_MAPPING_HOUSEHOLD_WASTE['recyclable']:
                label = 0  # Recyclable
            elif category in CATEGORY_MAPPING_HOUSEHOLD_WASTE['compostable']:
                label = 1  # Compostable
            elif category in CATEGORY_MAPPING_HOUSEHOLD_WASTE['landfill']:
                label = 2  # Landfill
            else:
                continue
            category_path = os.path.join(household_waste_path, category)
            if os.path.isdir(category_path):
                # Handle both real_world and default subdirectories
                for subdir in ['real_world', 'default']:
                    subdir_path = os.path.join(category_path, subdir)
                    if os.path.isdir(subdir_path):
                        for img_name in os.listdir(subdir_path):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                additional_waste_data.append((os.path.join(subdir_path, img_name), label))
                
       

        # Load garbage dataset
        if os.path.exists(garbage_dataset_path):
            for category in os.listdir(garbage_dataset_path):
                # Skip battery category
                if category == 'battery':
                    continue
                if category in CATEGORY_MAPPING_GARBAGE_DATASET['recyclable']:
                    if use_only_landfill_from_new:
                        continue
                    label = 0  # Recyclable
                elif category in CATEGORY_MAPPING_GARBAGE_DATASET['compostable']:
                    if use_only_landfill_from_new:
                        continue
                    label = 1  # Compostable
                elif category in CATEGORY_MAPPING_GARBAGE_DATASET['landfill']:
                    label = 2  # Landfill
                else:
                    continue
                category_path = os.path.join(garbage_dataset_path, category)
                if os.path.isdir(category_path):
                    for img_name in os.listdir(category_path):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            additional_waste_data.append((os.path.join(category_path, img_name), label))
                            
        

        # Load real waste dataset
        if os.path.exists(realwaste_path):
            for category in os.listdir(realwaste_path):
                if category in CATEGORY_MAPPING_REALWASTE['recyclable']:
                    if use_only_landfill_from_new:
                        continue
                    label = 0  # Recyclable
                elif category in CATEGORY_MAPPING_REALWASTE['compostable']:
                    if use_only_landfill_from_new:
                        continue
                    label = 1  # Compostable
                elif category in CATEGORY_MAPPING_REALWASTE['landfill']:
                    label = 2  # Landfill
                else:
                    continue
                category_path = os.path.join(realwaste_path, category)
                if os.path.isdir(category_path):
                    for img_name in os.listdir(category_path):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            additional_waste_data.append((os.path.join(category_path, img_name), label))
                            
        

        # Randomly split household data 80/20
        np.random.seed(25)
        np.random.shuffle(additional_waste_data)
        split_idx = int(len(additional_waste_data) * 0.8)
        if train:
            self.data.extend(additional_waste_data[:split_idx])
        else:
            self.data.extend(additional_waste_data[split_idx:])
        # Shuffle the combined data
        np.random.shuffle(self.data)
        
        # Print dataset statistics
        class_counts = {0: 0, 1: 0, 2: 0}
        for _, label in self.data:
            class_counts[label] += 1
        print(f"\nDataset statistics for {'training' if train else 'validation'} set:")
        print(f"Total images: {len(self.data)}")
        print(f"Recyclable (0): {class_counts[0]}")
        print(f"Compostable (1): {class_counts[1]}")
        print(f"Landfill (2): {class_counts[2]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a random different image as fallback
            return self.__getitem__((idx + 1) % len(self.data))

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 