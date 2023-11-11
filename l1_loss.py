# imports
import torch
import torch.nn as nn

# input_data: [1, -1] target_data: [0, 0]
# none: [1, 1]
# sum: [2]
# mean: [1]

# from torch
l1_torch_none = nn.L1Loss(reduction='none')
l1_torch_sum = nn.L1Loss(reduction='sum')
l1_torch_mean = nn.L1Loss(reduction='mean')

# my implementaion
def l1_loss(input_data, target_data, reduction='mean'):
    abs_difference = torch.abs(input_data-target_data)
    if reduction=='none':
        return abs_difference
    if reduction=='sum':
        return torch.sum(abs_difference)
    else:
        return torch.mean(abs_difference)

# initialize two vectors
input_data = torch.randn(2, 2)
target_data = torch.randn(2, 2)

# calculate the loss
loss_torch_none = l1_torch_none(input_data, target_data)
loss_my_none = l1_loss(input_data, target_data, reduction='none')
print(loss_torch_none)
print(loss_my_none)
print(loss_torch_none==loss_my_none)

loss_torch_sum = l1_torch_sum(input_data, target_data)
loss_my_sum = l1_loss(input_data, target_data, reduction='sum')
print(loss_torch_sum)
print(loss_my_sum.shape)
print(loss_torch_sum==loss_my_sum)





