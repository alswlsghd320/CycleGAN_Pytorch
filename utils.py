# 기환
# LR 및 ReplayBuffer 내용 구현

import random

import torch


# 리플레이 버퍼 구현 부분
# 리플레이 버퍼는 DQN에서 가져온 방식
# max_size만큼의 데이터를 보관하고 있다가 max_size를 초과하면 알부 기존 데이터를 내보내고 새로운 데이터를 삽입한다.
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


# reference: https://gaussian37.github.io/dl-pytorch-lr_scheduler/#lambdalr-1
# learning rate scheduler
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    # 처음에는 1에 근접한 값을 반환, epoch이 진행 될 수록 분모의 값이 작아져 0에 근접한 learning rate를 반환한다.
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
