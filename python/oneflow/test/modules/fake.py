import oneflow as flow
import torch
import numpy as np

batch = 1
channel = 2
height = 6
width = 6
device = "cuda"

weight_numpy = np.random.randn(channel).astype(np.float32)
bias_numpy = np.random.randn(channel).astype(np.float32)
fused_x = np.random.randn(batch, channel, height, width).astype(np.float32)
fused_x_tensor = flow.Tensor(fused_x).to(device)
fused_x_tensor.requires_grad = True

fused_weight_tensor = flow.nn.Parameter(flow.tensor(weight_numpy).to(device))
fused_bias_tensor = flow.nn.Parameter(flow.tensor(bias_numpy).to(device))

# fused_bn = flow.nn.FusedBatchNorm2d(channel).to(device)
# fused_bn.weight = fused_weight_tensor
# fused_bn.bias = fused_bias_tensor
# fused_out = fused_bn(fused_x_tensor, None)
fused_bn = flow.nn.BatchNorm2d(channel).to(device)
fused_bn.weight = fused_weight_tensor
fused_bn.bias = fused_bias_tensor

fused_out = fused_bn(fused_x_tensor)
fused_out = flow.nn.functional.relu(fused_out)


origin_x_tensor = torch.tensor(fused_x).to(device)
origin_x_tensor.requires_grad = True

origin_weight_tensor = torch.nn.Parameter(torch.tensor(weight_numpy).to(device))
origin_bias_tensor = torch.nn.Parameter(torch.tensor(bias_numpy).to(device))

origin_batch_norm = torch.nn.BatchNorm2d(channel).to(device)
origin_batch_norm.weight = origin_weight_tensor
origin_batch_norm.bias = origin_bias_tensor

origin_out = origin_batch_norm(origin_x_tensor)
origin_out = torch.nn.functional.relu(origin_out)

fused_out.sum().backward()
origin_out.sum().backward()

print()
print(fused_out)
print(origin_out)
print("+++++++++++++++++++++++++++++++++++++++++++++++")
# test output.
print(np.allclose(fused_out.numpy(), origin_out.detach().cpu().numpy(), atol=1e-4, rtol=1e-4))

print(fused_x_tensor.grad.numpy())
print(origin_x_tensor.grad.cpu().numpy())
# test input grad.
print(np.allclose(
        fused_x_tensor.grad.numpy(),
        origin_x_tensor.grad.cpu().numpy(),
        atol=1e-4,
        rtol=1e-4,
))