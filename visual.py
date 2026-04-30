import torch
import torch.nn as nn
from torchviz import make_dot

# Define a simple model with one Linear layer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 32)  # 2 inputs → 32 outputs

    def forward(self, x):
        return self.fc(x)

import torch
import torch.nn as nn

model = SimpleModel()
x = torch.randn(1, 2)

# Einfach die Struktur printen
print(model)

# Oder mit torchinfo (pip install torchinfo) — braucht kein Graphviz
from torchinfo import summary
summary(model, input_size=(1, 2))
