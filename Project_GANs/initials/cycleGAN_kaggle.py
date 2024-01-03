import numpy as np
import torch
import pytorch_lightning as L

from utils_kaggle import plot_images


class Downsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 stride=2, padding=1, norm=True, lrelu=True):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels,
                                                         kernel_size=kernel_size, stride=stride,
                                                         padding=padding, bias=not norm))
        if norm:
            self.block.append(torch.nn.InstanceNorm2d(out_channels, affine=True))
        if lrelu is True:
            self.block.append(torch.nn.LeakyReLU(0.2, True) if lrelu else torch.nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Upsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 stride=2, padding=1, dropout=False):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, out_channels, bias=False,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                         torch.nn.InstanceNorm2d(out_channels, affine=True))
        if dropout:
            self.block.append(torch.nn.Dropout(0.5))
        self.block.append(torch.nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class UNetGenerator(torch.nn.Module):
    def __init__(self, hid_channels, in_channels=3, out_channels=3):
        super().__init__()
        self.downsampling_path = torch.nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False),  # 64x128x128
            Downsampling(hid_channels, hid_channels * 2),  # 128x64x64
            Downsampling(hid_channels * 2, hid_channels * 4),  # 256x32x32
            Downsampling(hid_channels * 4, hid_channels * 8),  # 512x16x16
            Downsampling(hid_channels * 8, hid_channels * 8),  # 512x8x8
            Downsampling(hid_channels * 8, hid_channels * 8),  # 512x4x4
            Downsampling(hid_channels * 8, hid_channels * 8),  # 512x2x2
            Downsampling(hid_channels * 8, hid_channels * 8, norm=False),  # 512x1x1, instance norm does not work on 1x1
        )
        self.upsampling_path = torch.nn.Sequential(
            Upsampling(hid_channels * 8, hid_channels * 8, dropout=True),  # (512+512)x2x2
            Upsampling(hid_channels * 16, hid_channels * 8, dropout=True),  # (512+512)x4x4
            Upsampling(hid_channels * 16, hid_channels * 8, dropout=True),  # (512+512)x8x8
            Upsampling(hid_channels * 16, hid_channels * 8),  # (512+512)x16x16
            Upsampling(hid_channels * 16, hid_channels * 4),  # (256+256)x32x32
            Upsampling(hid_channels * 8, hid_channels * 2),  # (128+128)x64x64
            Upsampling(hid_channels * 4, hid_channels),  # (64+64)x128x128
        )
        self.feature_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hid_channels * 2, out_channels,
                                     kernel_size=4, stride=2, padding=1),  # 3x256x256
            torch.nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for down in self.downsampling_path:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(self.upsampling_path, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x)


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels=3):
        super().__init__()
        self.block = torch.nn.Sequential(
            Downsampling(in_channels, hidden_channels, norm=False),
            Downsampling(hidden_channels, hidden_channels * 2),
            Downsampling(hidden_channels * 2, hidden_channels * 4),
            Downsampling(hidden_channels * 4, hidden_channels * 8, stride=1),
            torch.nn.Conv2d(hidden_channels * 8, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ImageBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            self.capacity = 0
            self.buffer = []

    def __call__(self, images):
        if self.buffer_size <= 0:  # non-used buffer
            return images
        returned_images = []
        for image in images:
            image = image.unsqueeze(dim=0)
            # fill buffer to max capacity
            if self.capacity < self.buffer_size:
                self.capacity += 1
                self.buffer.append(image)
                returned_images.append(image)
            else:
                p = np.random.uniform(low=0., high=1.)
                if p > 0.5:
                    index = np.random.randint(low=0, high=self.buffer_size)
                    temp = self.buffer[index].clone()
                    self.buffer[index] = image
                    returned_images.append(temp)
                else:
                    returned_images.append(image)
        return torch.cat(returned_images, dim=0)


class CycleGAN(L.LightningModule):
    def __init__(self, hidden_channels, optimizer, lr, betas,
                 lambda_idt, lambda_cycle, buffer_size,
                 num_epochs, decay_epochs):
        super().__init__()
        self.recon_P = None
        self.recon_M = None
        self.idt_P = None
        self.idt_M = None
        self.fake_P = None
        self.fake_M = None
        self.real_P = None
        self.real_M = None
        self.save_hyperparameters(ignore=["optimizer"])
        self.optimizer = optimizer
        self.automatic_optimization = False
        self.gen_PM = UNetGenerator(hid_channels=hidden_channels)
        self.gen_MP = UNetGenerator(hid_channels=hidden_channels)
        self.disc_M = Discriminator(hidden_channels=hidden_channels)
        self.disc_P = Discriminator(hidden_channels=hidden_channels)
        self.buffer_fake_M = ImageBuffer(buffer_size)
        self.buffer_fake_P = ImageBuffer(buffer_size)

    def forward(self, image):
        return self.gen_PM(image)

    def init_weights(self):
        def init_fn(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d)):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        for net in [self.gen_PM, self.gen_MP, self.disc_M, self.disc_P]:
            net.apply(init_fn)

    def setup(self, stage):
        if stage == "fit":
            self.init_weights()
            print("Model initialized.")

    def get_lr_scheduler(self, optimizer):
        def lr_lambda(epoch):
            len_decay_phase = self.hparams.num_epochs - self.hparams.decay_epochs + 1.0
            current_decay_step = max(0, epoch - self.hparams.decay_epochs + 1.0)
            val = 1.0 - current_decay_step / len_decay_phase
            return max(0.0, val)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        opt_config = {
            "lr": self.hparams.lr,
            "betas": self.hparams.betas,
        }
        opt_gen = self.optimizer(list(self.gen_PM.parameters()) + list(self.gen_MP.parameters()), **opt_config)
        opt_disc = self.optimizer(list(self.disc_M.parameters()) + list(self.disc_P.parameters()), **opt_config)
        optimizers = [opt_gen, opt_disc]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
        return optimizers, schedulers

    def adv_criterion(self, y_hat, y):
        return torch.nn.functional.mse_loss(y_hat, y)

    def recon_criterion(self, y_hat, y):
        return torch.nn.functional.l1_loss(y_hat, y)

    def get_adv_loss(self, fake, discriminate):
        fake_hat = discriminate(fake)
        real_labels = torch.ones_like(fake_hat)
        adv_loss = self.adv_criterion(fake_hat, real_labels)
        return adv_loss

    def get_idt_loss(self, real, idt, lambda_cycle):
        idt_loss = self.recon_criterion(idt, real)
        return self.hparams.lambda_idt * lambda_cycle * idt_loss

    def get_cycle_loss(self, real, recon, lambda_cycle):
        cycle_loss = self.recon_criterion(recon, real)
        return lambda_cycle * cycle_loss

    def get_gen_loss(self):
        adv_loss_PM = self.get_adv_loss(self.fake_M, self.disc_M)
        adv_loss_MP = self.get_adv_loss(self.fake_P, self.disc_P)
        total_adv_loss = adv_loss_PM + adv_loss_MP

        lambda_cycle = self.hparams.lambda_cycle
        idt_loss_MM = self.get_idt_loss(self.real_M, self.idt_M, lambda_cycle[0])
        idt_loss_PP = self.get_idt_loss(self.real_P, self.idt_P, lambda_cycle[1])
        total_idt_loss = idt_loss_MM + idt_loss_PP

        cycle_loss_MPM = self.get_cycle_loss(self.real_M, self.recon_M, lambda_cycle[0])
        cycle_loss_PMP = self.get_cycle_loss(self.real_P, self.recon_P, lambda_cycle[1])
        total_cycle_loss = cycle_loss_MPM + cycle_loss_PMP

        gen_loss = total_adv_loss + total_idt_loss + total_cycle_loss
        return gen_loss

    def get_disc_loss(self, real, fake, disc):
        real_hat = disc(real)
        real_labels = torch.ones_like(real_hat)
        real_loss = self.adv_criterion(real_hat, real_labels)

        fake_hat = disc(fake.detach())
        fake_labels = torch.zeros_like(fake_hat)
        fake_loss = self.adv_criterion(fake_hat, fake_labels)

        disc_loss = (fake_loss + real_loss) * 0.5
        return disc_loss

    def get_disc_loss_M(self):
        fake_M = self.buffer_fake_M(self.fake_M)
        return self.get_disc_loss(self.real_M, fake_M, self.disc_M)

    def get_disc_loss_P(self):
        fake_P = self.buffer_fake_P(self.fake_P)
        return self.get_disc_loss(self.real_P, fake_P, self.disc_P)

    def training_step(self, batch, batch_idx):
        self.real_M = batch["monet"]
        self.real_P = batch["photo"]
        opt_gen, opt_disc = self.optimizers()

        self.fake_M = self.gen_PM(self.real_P)
        self.fake_P = self.gen_MP(self.real_M)

        self.idt_M = self.gen_PM(self.real_M)
        self.idt_P = self.gen_MP(self.real_P)

        self.recon_M = self.gen_PM(self.fake_P)
        self.recon_P = self.gen_MP(self.fake_M)

        self.toggle_optimizer(opt_gen)
        gen_loss = self.get_gen_loss()
        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()
        self.untoggle_optimizer(opt_gen)

        self.toggle_optimizer(opt_disc)
        disc_loss_M = self.get_disc_loss_M()
        disc_loss_P = self.get_disc_loss_P()
        opt_disc.zero_grad()
        self.manual_backward(disc_loss_M)
        self.manual_backward(disc_loss_P)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        metrics = {
            "gen_loss": gen_loss,
            "disc_loss_M": disc_loss_M,
            "disc_loss_P": disc_loss_P
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_index):
        self.display_results(batch, batch_index, "validate")

    def test_step(self, batch, batch_index):
        self.display_results(batch, batch_index, "test")

    def predict_step(self, batch, batch_index):
        return self(batch)

    def display_results(self, batch, batch_index, step):
        real_P = batch
        fake_M = self(real_P)

        if step == "validate":
            title = f"Epoch {self.current_epoch + 1}: Photo-to-Monet Translation"
        else:
            title = f"Sample {batch_index + 1}: Photo-to-Monet Translation"

        plot_images(torch.cat([real_P, fake_M], dim=0), lines=len(real_P), title=title)

    def on_train_epoch_start(self):
        current_lr = self.lr_schedulers()[0].get_last_lr()[0]
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()
        logged_values = self.trainer.progress_bar_metrics
        print(
            f"Epoch {self.current_epoch + 1}",
            *[f"{k}: {v:.5f}" for k, v in logged_values.items()],
            sep=" - ",
        )

    def on_train_end(self):
        print("Training ended.")

    def on_predict_epoch_end(self):
        predictions = self.trainer.predict_loop.predictions
        num_batches = len(predictions)
        batch_size = predictions[0].shape[0]
        last_batch_diff = batch_size - predictions[-1].shape[0]
        print(f"Number of images generated: {num_batches * batch_size - last_batch_diff}")
