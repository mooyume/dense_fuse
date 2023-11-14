# -*- coding: utf-8 -*-

import torch

from densefuse_net import DenseFuseNet
from utils import test

device = 'cuda'

model = DenseFuseNet().to(device)
model.load_state_dict(torch.load('./train_result/model_weight.pkl')['weight'])

test_path = './images/IV_images/'     
test(test_path, model, mode='add')