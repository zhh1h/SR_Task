import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
from tqdm import tqdm




class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)  # 假设输入为单通道图像
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def is_pure_color_image(image):
    # Check if the images are solid

    return np.ptp(image) == 0

model = SRCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load hr data
hr_data = scipy.io.loadmat('originalSlices.mat')
# load hr data
lr_data = scipy.io.loadmat('blurredSlices.mat')
#print(hr_data.keys())
#print(lr_data.keys())
#print("Header:", hr_data['originalSlices'])
#print("Header:", lr_data['blurredSlices'])


original_slices_data = hr_data['originalSlices']
# print("Shape of 'originalSlices':", original_slices_data.shape)
# print("Dtype of 'originalSlices':", original_slices_data.dtype)
#original_slices = original_slices_data['image'][0]
blurred_slices_data = lr_data['blurredSlices']
# print("Shape of 'originalSlices':", original_slices.shape)
# print("Dtype of 'originalSlices':", original_slices.dtype)
hr= []
hr_num_images = original_slices_data['image'][0].shape[0]
#print(hr_num_images)

for i in range(hr_num_images):
    image = original_slices_data['image'][0, i]
    #filter images
    if not is_pure_color_image(image):
        hr.append(image)
print(f"Number of images in the list: {len(hr)}")



lr= []
lr_num_images = blurred_slices_data['image'][0].shape[0]
for i in range(lr_num_images):
    image = blurred_slices_data['image'][0, i]
    #filter image
    if not is_pure_color_image(image):
        lr.append(image)

print(f"Number of images in the list: {len(lr)}")

# plt.imshow(hr[200], cmap='gray')
#
# plt.show()
#
# plt.imshow(lr[200], cmap='gray')
#
# plt.show()

# plt.imshow(original_slices_data['image'][0, 105], cmap='gray')
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(lr, hr, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16,shuffle=True)


# training starts
num_epochs = 30
for epoch in range (num_epochs):
    model.train()
    for data in train_loader:
        inputs,labels = data[0],data[1]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluating performance on test dataset
model.eval()
high_psnr_count = 0
high_ssim_count = 0
total_count = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        outputs = model(inputs)
        for i in range(outputs.size(0)):
            output = outputs[i].squeeze().cpu().numpy()
            label = labels[i].squeeze().cpu().numpy()

            psnr = compare_psnr(label, output, data_range=1)
            ssim = compare_ssim(label, output, data_range=1)
            print(f"Image {total_count + 1}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

            if psnr > 50:
                high_psnr_count += 1
            if ssim > 0.75:
                high_ssim_count += 1

            total_count += 1

    psnr_ratio = high_psnr_count / total_count
    ssim_ratio = high_ssim_count / total_count

    print(f"PSNR > 50 dB Ratio: {psnr_ratio * 100:.2f}%")
    print(f"SSIM > 0.75 Ratio: {ssim_ratio * 100:.2f}%")

# randomly pick one image to present
random_batch = next(iter(test_loader))
inputs, labels = random_batch[0], random_batch[1]
idx = random.randint(0, inputs.shape[0] - 1)
selected_input = inputs[idx]
selected_label = labels[idx]

with torch.no_grad():
    output = model(selected_input)
    output_np = output.squeeze().detach().cpu().numpy()  # 将输出转换为numpy数组
    output_np_resized = resize(output_np, selected_label.shape, mode='constant', anti_aliasing=True)

    selected_input_np = selected_input.squeeze().cpu().numpy()
    selected_label_np = selected_label.cpu().numpy()




# Show the oringinal lr image, reconstruct hr image and original hr image
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(selected_input.squeeze().numpy(), cmap='gray')
axes[0].set_title('Original LR Image')
axes[1].imshow(output.squeeze().detach().cpu().numpy(), cmap='gray')
axes[1].set_title('Reconstructed HR Image')
axes[2].imshow(selected_label.squeeze().numpy(), cmap='gray')
axes[2].set_title('True HR Image')
for ax in axes:
    ax.axis('off')
plt.show()
print(f"selected PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

