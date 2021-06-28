# CycleGAN_Pytorch
## Contributor

* JinHong Min, github: [alswlsghd320](https://github.com/alswlsghd320)
* ChaeYoung Yoon, github: [chaeyeongyoon](https://github.com/chaeyeongyoon)
* KiHwan Kim, github: [luceinaltis](https://github.com/luceinaltis)

## Installation

```
git clone https://github.com/alswlsghd320/CycleGAN_Pytorch.git
cd CycleGAN_Pytorch


```
## Train & Test
```
#Train
python train.py --epoch 0 
                --n_epochs 100
                --dataset_path '.'
                --pretrained_path None
                --batch_size 8
                ...
                
"--epoch", type=int, default=0, help="epoch to start training from"
"--n_epochs", type=int, default=100, help="number of epochs of training"
"--dataset_path", type=str, default='.', help="path of the dataset"
"--pretrained_path", type=str, default=None, help="folder path of pretrained model(.pth)"
"--batch_size", type=int, default=8, help="size of the batches"
"--num_workers", type=int, default=4, help="the number of cpu workers"
"--lr", type=float, default=0.0002, help="adam: learning rate"
"--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient"
"--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient"
"--decay_epoch", type=int, default=10, help="epoch from which to start lr decay"
"--img_height", type=int, default=256, help="size of image height"
"--img_width", type=int, default=256, help="size of image width"
"--channels", type=int, default=3, help="number of image channels"
"--is_cuda", type=bool, default=True, help="whether to use cuda or not"
"--instance_init", type=bool, default=True, help="whether to initialize instance normalization or not"
"--sample_interval", type=int, default=500, help="interval between sampling model outputs"
"--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator"
"--lambda_cyc", type=float, default=10.0, help="cycle loss weight"
"--use_identity", type=bool, default=False, help="You can use identity loss for photo painting"
"--lambda_id", type=float, default=5.0, help="identity loss weight"
"--save_model", type=bool, default=True, help="save model"
"--save_interval", type=int, default=10, help="interval between saving model checkpoints"
'''

'''
#Test
python test.py # You must enter one of <image, folder, dataset_path>. 

"--image", type=str, default=None, help="image path"
"--folder", type=str, default=None, help="folder path"
"--dataset_path", type=str, default=None, help="path of the dataset"
"--model_path", type=str, default='.', help="path of the dataset"
"--img_height", type=int, default=256, help="size of image height"
"--img_width", type=int, default=256, help="size of image width"
"--channels", type=int, default=3, help="number of image channels"
"--is_cuda", type=bool, default=True, help="whether to use cuda or not"
"--instance_init", type=bool, default=True, help="whether to initialize instance normalization or not"
"--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator"
```

## Citation
https://arxiv.org/abs/1703.10593
```
@article{DBLP:journals/corr/ZhuPIE17,
  author    = {Jun{-}Yan Zhu and
               Taesung Park and
               Phillip Isola and
               Alexei A. Efros},
  title     = {Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
               Networks},
  journal   = {CoRR},
  volume    = {abs/1703.10593},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.10593},
  archivePrefix = {arXiv},
  eprint    = {1703.10593},
  timestamp = {Mon, 13 Aug 2018 16:48:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/ZhuPIE17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
