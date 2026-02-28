import torch
import matplotlib.pyplot as plt
import os

# 1. Load the time capsule
checkpoint = torch.load('checkpoint_best.pth', map_location='cpu')
metrics = checkpoint['metrics']

# 2. Extract the data
iters = metrics['iters']
train_loss = metrics['train_loss']
val_loss = metrics['val_loss']

# 3. Plot
plt.figure(figsize=(10, 6))
plt.plot(iters, train_loss, label='Train Loss', color='blue', alpha=0.8)
plt.plot(iters, val_loss, label='Validation Loss', color='red', alpha=0.8)
plt.title('Training Curve')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

save_location = os.path.join(os.getcwd(), 'plots', 'trainvalloss.png')
plt.savefig(save_location)

plt.show()

