import os
from datetime import datetime as dt
import tqdm
import argparse
import itertools
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import ImageDataset
from model import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, fix_seed_torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_path", type=str, default='.', help="path of the dataset")
parser.add_argument("--pretrained_path", type=str, default=None, help="folder path of pretrained model(.pth)")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--num_workers", type=int, default=4, help="the number of cpu workers")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--is_cuda", type=bool, default=True, help="whether to use cuda or not")
parser.add_argument("--instance_init", type=bool, default=True, help="whether to initialize instance normalization or not")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling model outputs")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--use_identity", type=bool, default=False, help="You can use identity loss for photo painting")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--save_model", type=bool, default=True, help="save model")
parser.add_argument("--save_interval", type=int, default=10, help="interval between saving model checkpoints")

args = parser.parse_args()

# Fix seed
fix_seed_torch(42)

def sample_images(imgs, G, F, device):
    G.eval()
    F.eval()

    with torch.no_grad():
        real_A = imgs['A'].to(device)
        fake_B = G(real_A)
        real_B = imgs['B'].to(device)
        fake_A = F(real_B)

        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)

        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    return image_grid

def train():
    assert args.dataset_path is not None

    device = 'cuda' if torch.cuda.is_available() and args.is_cuda else 'cpu'

    G = Generator(in_channels=args.channels, num_residual_blocks=args.n_residual_blocks, instance_norm_init=args.instance_init).to(device)
    F = Generator(in_channels=args.channels, num_residual_blocks=args.n_residual_blocks, instance_norm_init=args.instance_init).to(device)
    D_A = Discriminator(in_channels=args.channels, instance_norm_init=args.instance_init).to(device)
    D_B = Discriminator(in_channels=args.channels, instance_norm_init=args.instance_init).to(device)

    if args.pretrained_path is not None:
        # load model's weight dict
        G_dict = torch.load(os.path.join(args.pretrained_path, 'G.pth'), map_location=device)
        F_dict = torch.load(os.path.join(args.pretrained_path, 'F.pth'), map_location=device)
        D_A_dict = torch.load(os.path.join(args.pretrained_path, 'D_A.pth'), map_location=device)
        D_B_dict = torch.load(os.path.join(args.pretrained_path, 'D_B.pth'), map_location=device)

        # Models load pretrained state_dict
        G.load_state_dict(G_dict)
        F.load_state_dict(F_dict)
        D_A.load_state_dict(D_A_dict)
        D_B.load_state_dict(D_B_dict)

    # Define Tensorboard
    t = dt.today().strftime("%Y%m%d%H%M")
    tensorboard_path = os.path.join('run', t)
    writer = SummaryWriter(tensorboard_path)

    # Define Loss function
    GAN_loss = nn.MSELoss()
    Cycle_loss = nn.L1Loss()

    if args.use_identity:
        Identity_loss = nn.L1Loss()

    # Define Optimizers
    opt_GF = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=args.lr, betas=(args.b1, args.b2))
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # 4. Implementation - Training details - Second :
    # To reduce model oscillation, they update discriminators using a history of generated images rather than the ones produced by the latest generators.
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Define learning rate scheduler
    lr_scheduler_GF = torch.optim.lr_scheduler.LambdaLR(opt_GF,
                                                        lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(opt_D_A,
                                                        lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(opt_D_B,
                                                        lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

    train_transforms_ = [
        transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((args.img_height, args.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    val_transforms_ = [
        transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_ds = ImageDataset('female', 'male', mode='train', transforms_=train_transforms_)
    val_ds = ImageDataset('female', 'male', mode='valid', transforms_=val_transforms_)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=5, shuffle=True, num_workers=args.num_workers)

    output_shape = D_A.output_shape(torch.rand(args.batch_size, args.channels, args.img_height, args.img_width).to(device))

    iter_num = 0

    for epoc in range(args.epoch, args.n_epochs):
        for i, images in enumerate(tqdm.tqdm(train_dl, total=len(train_dl), mininterval=0.01)):
            real_A = images['A'].to(device) # Real Female
            real_B = images['B'].to(device) # Real Male

            # Ground truths
            valid = torch.ones(output_shape, requires_grad=False).to(device)
            fake = torch.zeros(output_shape, requires_grad=False).to(device)

            # Train G,F
            G.train()
            F.train()

            opt_GF.zero_grad()

            # GAN Loss
            fake_B = G(real_A) # A => B'
            fake_A = F(real_B) # B => A'

            loss_GAN_G = GAN_loss(D_B(fake_B), valid)
            loss_GAN_F = GAN_loss(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_G + loss_GAN_F) / 2
            writer.add_scalar('GAN loss', loss_GAN.item(), iter_num)

            # Cycle Loss
            recon_A = F(fake_B) # A => B' => A"
            recon_B = G(fake_A) # B => A' => B"

            loss_cycle_A = Cycle_loss(recon_A, real_A) #||F(G(A)) - A||
            loss_cycle_B = Cycle_loss(recon_B, real_B) #||G(F(B)) - B||

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            writer.add_scalar('Cycle loss', loss_cycle.item(), iter_num)

            if args.use_identity:
                loss_id_A = Identity_loss(F(real_A), real_A) #||F(A) - A||
                loss_id_B = Identity_loss(G(real_B), real_B) #||G(B) - B||
                loss_id = (loss_id_A + loss_id_B) / 2
                writer.add_scalar('Identity loss', loss_id.item(), iter_num)

                total_GF_loss = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_id

            else:
                total_GF_loss = loss_GAN + args.lambda_cyc * loss_cycle

            writer.add_scalar('Total GF loss', total_GF_loss.item(), iter_num)
            total_GF_loss.backward()
            opt_GF.step()

            # Train D_A
            opt_D_A.zero_grad()

            # Real loss
            loss_real = GAN_loss(D_A(real_A), valid) #log DY(y)
            # Fake loss
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = GAN_loss(D_A(fake_A_.detach()), fake) # log(1 − DY(G(x))

            loss_D_A = (loss_real + loss_fake) / 2
            writer.add_scalar('D_A loss', loss_D_A.item(), iter_num)

            loss_D_A.backward()
            opt_D_A.step()

            # Train D_B
            opt_D_B.zero_grad()

            # Real loss
            loss_real = GAN_loss(D_B(real_B), valid)

            # Fake loss
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = GAN_loss(D_B(fake_B_.detach()), fake)

            loss_D_B = (loss_real + loss_fake) / 2
            writer.add_scalar('D_B loss', loss_D_B.item(), iter_num)

            loss_D_B.backward()
            opt_D_B.step()

            iter_num += 1

            if (iter_num+1) % args.sample_interval == 0:
                imgs = next(iter(val_dl))
                eval_output = sample_images(imgs, G, F, device=device)
                writer.add_image('output_image', eval_output, iter_num)

        # Update learning rates
        lr_scheduler_GF.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if args.save_model and (epoc + 1) % args.save_interval == 0:
            save_path = os.path.join('save_model', t)
            torch.save(G.state_dict(), f'{save_path}/G_{epoc}epochs')
            torch.save(F.state_dict(), f'{save_path}/F_{epoc}epochs')
            torch.save(D_A.state_dict(), f'{save_path}/D_A_{epoc}epochs')
            torch.save(D_B.state_dict(), f'{save_path}/D_B_{epoc}epochs')

    if args.save_model:
        torch.save(G.state_dict(), f'{save_path}/G.pth')
        torch.save(F.state_dict(), f'{save_path}/F.pth')
        torch.save(D_A.state_dict(), f'{save_path}/D_A.pth')
        torch.save(D_B.state_dict(), f'{save_path}/D_B.pth')

    writer.close()

if __name__ == '__main__':
    train()