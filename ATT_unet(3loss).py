import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

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

# Create datasets
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
num_workers = 0  # Increase this if you have a powerful CPU
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")  # Includes augmented samples
print(f"Test dataset size: {len(test_dataset)}")

import torch.nn as nn



class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class Model(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(Model, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 16)
        self.Conv2 = ConvBlock(16, 32)
        self.Conv3 = ConvBlock(32, 64)
        self.Conv4 = ConvBlock(64, 128)
        self.Conv5 = ConvBlock(128, 256)

        self.Up5 = UpConv(256, 128)
        self.Att5 = AttentionBlock(F_g=128, F_l=128, n_coefficients=65)
        self.UpConv5 = ConvBlock(256, 128)

        self.Up4 = UpConv(128, 64)
        self.Att4 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv4 = ConvBlock(128, 64)

        self.Up3 = UpConv(64, 32)
        self.Att3 = AttentionBlock(F_g=32, F_l=32, n_coefficients=16)
        self.UpConv3 = ConvBlock(64, 32)

        self.Up2 = UpConv(32, 16)
        self.Att2 = AttentionBlock(F_g=16, F_l=16, n_coefficients=8)
        self.UpConv2 = ConvBlock(32, 16)

        self.Conv = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)

    ######################################
    def init(self):
        self.load_state_dict(torch.load('model.pth',weights_only=True))
    ######################################

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)
        #print(f"encoder1 shape: {e1.size()}")
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        #print(f"encoder2 shape: {e2.size()}")
        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        #print(f"encoder3 shape: {e3.size()}")
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        #print(f"encoder4 shape: {e4.size()}")
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        #print(f"encoder5 shape: {e5.size()}")


        d5 = self.Up5(e5)
        d5 = F.pad(d5,(0,1,0,1), mode='replicate')
        #print(f"dec 5 shape before att with e4: {d5.size()}")
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        #print(f"dec 4 shape before att with e3: {d4.size()}")
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        #print(f"dec 3 shape before att with e2: {d3.size()}")
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        #print(f"dec 2 shape before att with e1: {d2.size()}")
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out 
		
model = Model().to('cuda')

image = next(iter(dataloaders['train']))[0].to('cuda')
out = model(image)
print(image.shape)
print(out.shape)


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

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        elif self.reduction == 'none':
            return dice_loss
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")


import torch.nn.functional as F

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
        smooth = 1e-6  # Small value to prevent division by zero
        aux_criterion = FocalLoss(alpha=0.75, gamma=2, reduction='mean')
        fl_lss = aux_criterion(output, target)
        # Normalize target to [0, 1]
        target = target / 255.0
        target = target.float()

        # Apply sigmoid activation to logits
        output = torch.sigmoid(output)

        # Flatten tensors for element-wise operations
        output = output.view(-1)
        target = target.view(-1)

        # Compute intersection and union
        intersection = (output * target).sum()
        union = output.sum() + target.sum()

        # Compute Dice coefficient
        dice_coeff = (2. * intersection + smooth) / (union + smooth)

        # Compute Dice loss
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




criterion = FocalLoss() #DiceFocalLoss #DiceLoss() 
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


model = model.to('cuda')  # Move model to GPU if available

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
    best_dice = 0
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
                inputs = inputs.to('cuda')
                masks = masks.to('cuda')

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
                    for ii in range(len(masks)):
                      print((torch.sum(masks[ii])>0) / (torch.sum(masks[ii])>=0))
                    loss = criterion(outputs,masks)
                    print("pred size is:", outputs.size(), "______ target size is:", masks.size())
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        dice_scores.append(dice_score(outputs, masks))
                        if dice_scores[-1] > best_dice:
                            best_dice == dice_scores[-1]
                            torch.save(model.state_dict(), 'att_unet_focal_best-model-parameters.pt')
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
                print(f'Dice score: {torch.tensor(dice_scores).mean()}')

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