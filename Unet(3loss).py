import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the split ratio and other parameters
test_size = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset function for teeth segmentation
class TeethSegmentationDataset(Dataset):
    def __init__(self, image_list: list, mask_list: list, transform: A.Compose, augment_transform: A.Compose = None):
        """
        Args:
            image_list (list): List of image file paths.
            mask_list (list): List of corresponding mask file paths.
            transform (A.Compose): Transformations for original data.
            augment_transform (A.Compose, optional): Transformations for augmented data. If None, no augmentation.
        """
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
        self.augment_transform = augment_transform

        # Determine total samples (original + augmented if enabled)
        self.total_samples = len(self.image_list)
        if augment_transform is not None:
            self.total_samples *= 3

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of image-mask pairs (original + augmented).
        """
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and its corresponding mask.
        """
        # Determine if this is an augmented sample
        is_augmented = index >= len(self.image_list)
        actual_index = index % len(self.image_list)

        # Load the image and mask
        image_path = self.image_list[actual_index]
        mask_path = self.mask_list[actual_index]
        image = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply augmentations
        if is_augmented and self.augment_transform:
            transformed = self.augment_transform(image=image, mask=mask)
        else:
            transformed = self.transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']  # Add channel dimension to the mask

        return image, mask.float()


# Define original and augmented transformations
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0),
    ToTensorV2(),
])

base_transform = A.Compose([
    A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0),
    ToTensorV2(),
])

# Get all images and masks
image_dir = '/.../S_Z_IMG'
mask_dir = '/.../S_Z_MSK'

all_images = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
all_masks = sorted([os.path.join(mask_dir, msk) for msk in os.listdir(mask_dir)])

# Split data into test and training
test_images = all_images[:test_size]
test_masks = all_masks[:test_size]
train_images = all_images[test_size:]
train_masks = all_masks[test_size:]


train_dataset = TeethSegmentationDataset(
    image_list=train_images,
    mask_list=train_masks,
    transform=base_transform,
    augment_transform=augmenter  # Include augmentations for training data
)

test_dataset = TeethSegmentationDataset(
    image_list=test_images,
    mask_list=test_masks,
    transform=base_transform  # No augmentations for test data
)

# Define DataLoaders
batch_size = 8
num_workers = 0  
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")  # Includes augmented samples
print(f"Test dataset size: {len(test_dataset)}")

import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        num_classes = 1
        # Encoder
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2,stride = 2),
        )

        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2,stride = 2)
        )
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2,stride = 2)
        )

        # Decoder
        self.decoder_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),)

        self.decoder_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=2, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=2,padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),)

        self.decoder_block4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=2, padding='same'),
            nn.Tanh(),
            )
        

    #######DO NOT CHANGE THIS PART########
    def init(self):
        self.load_state_dict(torch.load('model.pth',weights_only=True))
    ######################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method defines the forward pass of the model.

        Args:
            x (tensor): The input tensor, in the shape of (batch_size,1,512,512).

        Returns:
            mask (tensor): The output tensor logits, in the shape of (batch_size,1,512,512).
        """
        # Add you code here
        x1 = self.encoder_block1(x)
        #print("x1 size = ", x1.size())
        x2 = self.encoder_block2(x1)
        #print("x2 size = ", x2.size())
        x3 = self.encoder_block3(x2)
        #print("x3 size = ", x3.size())
        x4 = self.encoder_block4(x3)
        #print("x4 size = ", x4.size())

        y1 = self.decoder_block1(x4) 
        #print("y1 size = ", y1.size())
        y1 = torch.add(x3, y1)
        y2 = self.decoder_block2(y1)
        #print("y2 size = ", y2.size()) 
        y2 = torch.add(x2, y2)
        y3 = self.decoder_block3(y2)
        #print("y3 size = ", y3.size()) 
        y3 = torch.add(x1, y3)
        mask = self.decoder_block4(y3)

        return mask
		
model = Model().to(device)

image = next(iter(dataloaders['train']))[0].to(device)
out = model(image)
print(image.shape)
print(out.shape)
import math
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)
"""def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)"""
model.apply(weights_init_kaiming)

def dice_score(pred: torch.Tensor, target_mask: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Computes the Dice score between the predicted and target segmentation masks.

    Args:
        pred (torch.Tensor): The predicted mask tensor, with values in range [0, 1].
        target_one_target_maskhot (torch.Tensor): The ground truth mask.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        float: The Dice score, a similarity metric between 0 and 1.
    """
    pred = pred>0
    pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target_flat = target_mask.contiguous().view(target_mask.shape[0], target_mask.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)

    dice = (2. * intersection + epsilon) / (union + epsilon)

    dice_mean = dice.mean(dim=1)

    return dice_mean.mean()


import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Dice Loss for binary classification with optional reduction.

        Args:
            reduction (str): Specifies the reduction to apply: 'mean', 'sum', or 'none'.
        """
        super(DiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        """
        Compute the Dice Loss for binary classification.

        Args:
            output (Tensor): Model logits of shape (N,) or (N, 1).
            target (Tensor): Ground-truth labels of shape (N,) with values in [0, 255].

        Returns:
            Tensor: Computed Dice Loss.
        """
        smooth = 1e-6  

        
        target = target / 255.0
        target = target.float()

        
        output = torch.sigmoid(output)

        
        output = output.view(-1)
        target = target.view(-1)

        
        intersection = (output * target).sum()
        union = output.sum() + target.sum()

        
        dice_coeff = (2. * intersection + smooth) / (union + smooth)

        
        dice_loss = 1 - dice_coeff

        # Handle reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        elif self.reduction == 'none':
            return dice_loss
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")




class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        """
        Focal Loss for binary classification with optional alpha weighting.

        Args:
            gamma (float): Focusing parameter. Default is 2.
            alpha (float): Weighting factor for the positive class. Default is 0.25.
            reduction (str): Specifies the reduction to apply: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss for binary classification.

        Args:
            inputs (Tensor): Model logits of shape (N,) or (N, 1).
            targets (Tensor): Ground-truth labels of shape (N,) with values in [0, 255].

        Returns:
            Tensor: Computed Focal Loss.
        """
        
        targets = targets / 255.0
        targets = targets.float()

        
        inputs = torch.sigmoid(inputs).squeeze()

        
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        
        pt = torch.exp(-bce_loss)

        
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss *= alpha_factor

        # Apply the specified reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")



class DiceFocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Dice Loss for binary classification with optional reduction.

        Args:
            reduction (str): Specifies the reduction to apply: 'mean', 'sum', or 'none'.
        """
        super(DiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        """
        Compute the Dice Loss for binary classification.

        Args:
            output (Tensor): Model logits of shape (N,) or (N, 1).
            target (Tensor): Ground-truth labels of shape (N,) with values in [0, 255].

        Returns:
            Tensor: Computed Dice Loss.
        """
        smooth = 1e-6 
        aux_criterion = FocalLoss(alpha=0.75, gamma=2, reduction='mean')
        fl_lss = aux_criterion(output, target)
        
        target = target / 255.0
        target = target.float()

        
        output = torch.sigmoid(output)

       
        output = output.view(-1)
        target = target.view(-1)

        
        intersection = (output * target).sum()
        union = output.sum() + target.sum()

        dice_coeff = (2. * intersection + smooth) / (union + smooth)

       
        dice_focal_loss = 1 - dice_coeff + fl_lss
        
        # Handle reduction
        if self.reduction == 'mean':
            return dice_focal_loss.mean()
        elif self.reduction == 'sum':
            return dice_focal_loss.sum()
        elif self.reduction == 'none':
            return dice_focal_loss
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")

    
criterion = DiceLoss() #FocalLoss() #DiceFocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

import matplotlib.pyplot as plt
def visualize_mask(inputs: torch.Tensor, masks: torch.Tensor, outputs: torch.Tensor):
    # Convert tensors to numpy for visualization
    sample_index = 0  # Index of the sample to visualize
    channel = 0
    print(f'Dice score is {dice_score(outputs[sample_index:sample_index+1,channel:channel+1],masks[sample_index:sample_index+1,channel:channel+1])}')

    inputs_np = inputs.cpu().numpy()
    masks_np = masks.cpu().numpy()
    outputs_np = outputs.detach().cpu().numpy()

    # Choose a sample to visualize

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(inputs_np[sample_index].transpose(1, 2, 0), cmap='gray')  # Assuming inputs are in CxHxW format
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(masks_np[sample_index], cmap='gray')  # Display the first channel of the mask
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(outputs_np[sample_index], cmap='gray')  # Display the first channel of the output
    plt.title("Model Output Mask")
    plt.axis('off')

    plt.show()
	
	
import torch
import time
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim


model = model.to(device)  # Move model to GPU if available

# Training function with visualization support
def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 100
) -> nn.Module:

    since = time.time()

    train_losses = []
    test_losses = []
    dice_scores_epoch = []

    for epoch in range(num_epochs):
        dice_scores = []
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    outputs = outputs.squeeze(1)
                    #print('im', torch.max(outputs))
                    #print('im', torch.min(outputs))
                    #print('pred sum', torch.sum(outputs>0))
                    #print('outputs',torch.max(outputs))
                    #print('mask',torch.max(masks))
                    #print('mask sum', torch.sum(masks))
                    #print('mask sum', torch.sum(1-masks))
                    print("pred size is:", outputs.size(), "______ target size is:", masks.size())
                    print("Unique mask values:", torch.unique(masks))
                    for ii in range(len(masks)):
                      print((torch.sum(masks[ii])>0) / (torch.sum(masks[ii])>=0))
                    loss =  criterion(outputs,masks)
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        dice_scores.append(dice_score(outputs, masks))

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)
                dice_scores_epoch.append(torch.tensor(dice_scores).mean().item())

                visualize_mask(inputs, masks, outputs)


            print(f'{phase} Loss: {epoch_loss:.4f}')
            if phase == 'test':
                print(f'Dice score test: {torch.tensor(dice_scores).mean()}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Plot the results
    epochs_range = range(num_epochs)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, test_losses, label="Test Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Test Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, dice_scores_epoch, label="Dice Score", color="green")
    plt.legend(loc="lower right")
    plt.title("Dice Score")

    plt.show()

    return model

# Train the model
model = train_model(model, dataloaders, criterion, optimizer)

