import torch
import matplotlib.pyplot as plt
from src.models.gan import HandsGAN

hands_gan = HandsGAN.load_from_checkpoint(
    checkpoint_path='lightning_logs/version_0/checkpoints/epoch=49-step=20400.ckpt',
    hparams_file='lightning_logs/version_0/hparams.yaml',
    map_location='cpu'
)

latent_vector = torch.randn(8, 128, 1, 1)
images = hands_gan(latent_vector)


num_images = images.shape[0]

rows = 2
cols = 4

fig, axs = plt.subplots(int(images.shape[0]/4), int(images.shape[0]/2), figsize=(10, 5))

for i in range(images.shape[0]):
    img = images[i]
    img = img.to('cpu').detach().numpy().transpose(1, 2, 0)
    ax = axs[i // cols, i % cols]
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()
