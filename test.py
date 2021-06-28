import argparse
import os
from PIL import Image
import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None, help="image path")
parser.add_argument("--folder", type=str, default=None, help="folder path")
parser.add_argument("--dataset_path", type=str, default=None, help="path of the dataset")
parser.add_argument("--model_path", type=str, default='.', help="path of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--is_cuda", type=bool, default=True, help="whether to use cuda or not")
parser.add_argument("--instance_init", type=bool, default=True, help="whether to initialize instance normalization or not")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")

args = parser.parse_args()

def test():
    device = 'cuda' if torch.cuda.is_available() and args.is_cuda else 'cpu'

    G = Generator(in_channels=args.channels, num_residual_blocks=args.n_residual_blocks,
                  instance_norm_init=args.instance_init).to(device)
    F = Generator(in_channels=args.channels, num_residual_blocks=args.n_residual_blocks,
                  instance_norm_init=args.instance_init).to(device)

    G_dict = torch.load(os.path.join(args.model_path, 'G.pth'), map_location=device)
    F_dict = torch.load(os.path.join(args.model_path, 'F.pth'), map_location=device)

    G.load_state_dict(G_dict)
    F.load_state_dict(F_dict)

    test_transforms_ = [
        transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    if args.image is not None:
        filepath = args.image
        filename = os.path.basename(filepath)
        img = Image.open(filepath)
        img = img.convert('RGB') if args.channels == 3 or img.mode != "RGB" else img.convert('L')
        img = test_transforms_(img).to(device)

        with torch.no_grad():
            output_G = G(img)
            output_F = F(img)

        save_image(output_G, os.path.join(filepath, '..', f'{filename}_G'), normalize=False)
        save_image(output_F, os.path.join(filepath, '..', f'{filename}_F'), normalize=False)

    elif args.folder is not None:
        folder_path = args.folder
        folder_name = os.path.basename(folder_path)
        save_path = os.path.join(folder_path, '..', f'{folder_name}_synthetic')

        if os.path.isdir(save_path):
            os.mkdir(save_path)

        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            img = Image.open(img_path)
            img = img.convert('RGB') if args.channels == 3 or img.mode != "RGB" else img.convert('L')
            img = test_transforms_(img).to(device)

            with torch.no_grad():
                output_G = G(img)
            save_image(output_G, os.path.join(save_path, img), normalize=False)

    elif args.dataset_path is not None:
        test_ds = ImageDataset('female', 'male', mode='test', transforms_=test_transforms_)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        f_save_path = os.path.join('dataset', 'synthetic_F')
        g_save_path = os.path.join('dataset', 'synthetic_G')

        if os.path.isdir(f_save_path):
            os.mkdir(f_save_path)
        if os.path.isdir(g_save_path):
            os.mkdir(g_save_path)

        for i, images in enumerate(tqdm.tqdm(test_dl, total=len(test_dl), mininterval=0.01)):
            real_A = images['A'].to(device)  # Real Female
            real_B = images['B'].to(device)  # Real Male

            with torch.no_grad():
                fake_B = G(real_A)  # A => B'
                fake_A = F(real_B)  # B => A'

            fake_b_filename = os.path.basename(test_ds.files_A[i % len(test_ds.files_A)])
            fake_a_filename = os.path.basename(test_ds.files_B[i % len(test_ds.files_B)])

            save_image(fake_B, os.path.join(g_save_path, fake_b_filename), normalize=False)
            save_image(fake_A, os.path.join(f_save_path, fake_a_filename), normalize=False)

if __name__ == '__main__':
    test()
