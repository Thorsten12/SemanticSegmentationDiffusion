from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Block.
    
    Warum ist dieses MLP so wichtig?
    Während die Self-Attention-Schicht dafür zuständig ist, Informationen 
    ZWISCHEN verschiedenen Pixeln oder Sequenzpositionen auszutauschen 
    (räumlicher Kontext), arbeitet das MLP isoliert auf jedem einzelnen Pixel. 
    
    Es nimmt die durch die Attention neu gesammelten Informationen (die Features/Kanäle) 
    eines Pixels und verarbeitet diese tiefgreifend in sich selbst. Das MLP mischt 
    also nur entlang der Feature-Dimension. 
    
    Durch die Expansion der Kanäle (meist das 2- bis 4-fache in der Mitte) und 
    die nicht-lineare Aktivierungsfunktion (GELU) erhält das Modell hier den 
    nötigen "Denkraum", um komplexe Muster zu speichern und die gesammelten 
    Erkenntnisse der Attention zu festigen.
    
    Zusammenfassung der Aufgabenteilung im Transformer:
    - Attention = Informationsaustausch über das gesamte Bild (Kommunikation).
    - MLP = Tiefe Verarbeitung der gesammelten Informationen pro Pixel (Einzelarbeit).
    """
    def __init__(self, in_channels, mlp_ratio=4, mlp_p=0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels * mlp_ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(in_channels * mlp_ratio, in_channels)
        self.drop2 = nn.Dropout(mlp_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        x= self.fc2(x)
        x = self.drop2(x)
        return x
    
