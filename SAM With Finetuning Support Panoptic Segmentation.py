import os
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset as HFDataset  # Alias to avoid conflict
from transformers import SamProcessor
from torch.utils.data import Dataset as TorchDataset
from transformers import AdamW

def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if y_indices.size == 0 or x_indices.size == 0:
        return [0.0, 0.0, 0.0, 0.0], 0  # bbox for healthy plants
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    return [float(x_min), float(y_min), float(x_max), float(y_max)], 1

image_directory = "C:\\Users\\Stell\\Desktop\\DVA 309\\Images10"
mask_directory = "C:\\Users\\Stell\\Desktop\\DVA 309\\masks10"

image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]
mask_files = [os.path.join(mask_directory, file) for file in os.listdir(mask_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

images = [Image.fromarray(cv2.imread(file, cv2.IMREAD_UNCHANGED)) for file in image_files]
masks = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in mask_files]
bboxes, labels = zip(*[get_bounding_box(mask) for mask in masks])
masks_pil = [Image.fromarray(mask) for mask in masks]

dataset_dict = {
    "image": images,
    "label": masks_pil,
    "bbox": bboxes,
    "health": labels
}

# Convert to dataset
dataset = HFDataset.from_dict(dataset_dict)
print(f"Dataset created with {len(dataset)} entries.")

class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt, _ = get_bounding_box(ground_truth_mask)  # Get bounding box

        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset, processor=processor)

example = train_dataset[0]
for k, v in example.items():
    print(k, v.shape)


# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)


batch["ground_truth_mask"].shape

# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = AdamW(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0.01)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')


from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F
#Training loop


# Adjust the training loop to resize the ground truth masks
num_epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        
        ground_truth_masks_resized = F.interpolate(ground_truth_masks.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)

        loss = seg_loss(predicted_masks, ground_truth_masks_resized)

        
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

