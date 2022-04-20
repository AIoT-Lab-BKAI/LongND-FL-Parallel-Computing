

from models.vgg import make_layers_mnist, vgg11_pill

import torch
model = vgg11_pill(85)
a = model(torch.rand(10,3,224,224))
print(model)
breakpoint()
