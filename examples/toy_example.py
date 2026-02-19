import torch
from softdtw_cuda import SoftDTW

torch.manual_seed(0)

x = torch.randn(2, 50, 3, requires_grad=True, device="cuda")
y = torch.randn(2, 47, 3, device="cuda")

sdtw = SoftDTW(gamma=0.1)

out = sdtw(x, y)
print("out:", out)

out.sum().backward()
print("grad finite:", torch.isfinite(x.grad).all())
