import os
import glob
import requests
import sys
import zipfile
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, dataset_A: str, dataset_B: str, mode: str, transforms_=None, root_dir: str = os.getcwd()):
        """
            이미지 데이터셋을 불러옵니다.

            만약, 파일이 없다면 다운로드합니다.

            train:test:valid의 데이터셋 비율은 7:2:1 입니다.

            리턴값은 딕셔너리로 제공됩니다. ex) {'A': PIL 객체, 'B': PIL 객체}

            Parameters:
                `dataset_A`, `dataset_B`
                    `female` - female face sets

                    `male` - male face sets

                `mode` - can use among `train`, `test`, and `valid`.

                `transforms_` - sets of transforming

                `root_dir` - root directory to manipulate image dataset, default value is starting point of current process.
        """
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.mode = mode
        self.applier = transforms.Compose(transforms_)
        self.root_dir = root_dir

        self.__download(dataset_A, os.path.join(root_dir, 'datasets'))
        self.__download(dataset_B, os.path.join(root_dir, 'datasets'))

        self.files_A = sorted(glob.glob(os.path.join(
            root_dir, 'datasets', dataset_A, mode, '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(
            root_dir, 'datasets', dataset_B, mode, '*.*')))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        if image_A.mode != "RGB":
            image_A = self.__to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = self.__to_rgb(image_B)

        item_A = self.applier(image_A)
        item_B = self.applier(image_B)

        #print(item_A)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __download(self, dataset_name, dir):
        """
            Dataset Downloader

            `female` - Female's face set in CelebA

            `male` - Male's face set in CelebA
        """
        if os.path.isdir('datasets') is False:
            os.mkdir('datasets')

        db = {'female': 'https://aimldatasets.s3.ap-northeast-2.amazonaws.com/datasets/female.zip',
              'male': 'https://aimldatasets.s3.ap-northeast-2.amazonaws.com/datasets/male.zip'}
        remote_url = db[dataset_name]
        file_name = remote_url.split('/')[-1]

        if os.path.isfile(os.path.join('datasets', file_name)) is False:
            with requests.get(remote_url, stream=True) as r:
                r.raise_for_status()
                print(f'Getting dataset of \"{dataset_name}\"')
                total_length = int(r.headers.get('content-length'))
                with open(os.path.join("datasets", file_name), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        current_length = f.tell()
                        sys.stdout.write(
                            f'Download progress: {100 * current_length/total_length}%     \r')
                        sys.stdout.flush()

        if os.path.isdir(os.path.join('datasets', file_name.split('.')[0])) is False:
            sys.stdout.write(f'Extracting...\n')
            zip_file = zipfile.ZipFile(os.path.join(dir, file_name))
            zip_file.extractall(dir)

        if glob.iglob(os.path.join('datasets', file_name.split('.')[0], '*.*')):
            print(f'Spliting dataset of \"{dataset_name}\"')
            self.__distributor(dataset_name, os.path.join(
                dir, file_name.split('.')[0]))

        print(f'OK!')

    def __distributor(self, dataset_name, dir):
        if os.path.isdir(os.path.join(dir, 'train')) == False:
            os.mkdir(os.path.join(dir, 'train'))

        if os.path.isdir(os.path.join(dir, 'test')) == False:
            os.mkdir(os.path.join(dir, 'test'))

        if os.path.isdir(os.path.join(dir, 'valid')) == False:
            os.mkdir(os.path.join(dir, 'valid'))

        files = glob.glob(os.path.join(dir, '*.*'))

        for idx, val in enumerate(files):
            rand = random.randrange(1, 11)
            file_name = os.path.basename(val)
            if rand in range(1, 2):
                # valid
                new_file_dir = os.path.join(
                    os.path.dirname(val), 'valid', file_name)
                os.system(f'mv {val} {new_file_dir}')
            elif rand in range(2, 4):
                # test
                new_file_dir = os.path.join(
                    os.path.dirname(val), 'test', file_name)
                os.system(f'mv {val} {new_file_dir}')
            else:
                # train
                new_file_dir = os.path.join(
                    os.path.dirname(val), 'train', file_name)
                os.system(f'mv {val} {new_file_dir}')

            if idx % 1000 == 0:
                sys.stdout.write(
                    f'Spliting progress: {100 * idx/len(files)}%     \r')
                sys.stdout.flush()

    def __to_rgb(image):
        """
            RGB Converter
        """
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image


