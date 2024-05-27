import lightning
import matplotlib.pyplot as plt


class LossLoggerCallback(lightning.Callback):
    def __init__(self):
        super().__init__()
        self.generator_losses = []
        self.discriminator_losses = []

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        g_loss = trainer.logged_metrics['g_loss']
        d_loss = trainer.logged_metrics['d_loss']
        self.generator_losses.append(g_loss.item())
        self.discriminator_losses.append(d_loss.item())

def plot_gan_loss(
        epochs: int,
        g_loss: list,
        d_loss: list,
        save_dir: str,
        format: str='png'
):
    '''
    Plot the GAN loss over epochs and save the plot to a file.

    Parameters:
    - epochs (int): Number of epochs.
    - g_loss (Sequence): List of generator loss values.
    - d_loss (Sequence): List of discriminator loss values.
    - save_dir (str): Directory to save the plot.
    - format (str): Format to save the plot, default is 'png'.
    '''

    plt.figure(figsize=(10, 5))
    
    plt.plot(list(range(epochs)), g_loss, label='Generator Loss', color='blue')
    plt.plot(list(range(epochs)), d_loss, label='Discriminator Loss', color='red')
    
    plt.title('GAN Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.grid(True)
    
    save_path = f"{save_dir}/gan_loss.{format}"
    plt.savefig(save_path, format=format, dpi=300)
    plt.close()

    print(f"Plot saved to {save_path}")