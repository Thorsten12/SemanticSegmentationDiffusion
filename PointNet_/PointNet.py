import torch
import torch.nn as nn

# Dein bestehender Encoder (unverändert)
class PointNetEncoder2D(nn.Module):
    def __init__(self, global_feat_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, global_feat_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(global_feat_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.shape[-1] == 2:
            x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

# NEU: Der Decoder
class PointNetDecoder2D(nn.Module):
    def __init__(self, global_feat_dim=1024, num_points=200):
        super().__init__()
        self.num_points = num_points
        
        # Aus 1024 Features wieder 400 Zahlen (200 Punkte * 2 Koordinaten) machen
        self.fc1 = nn.Linear(global_feat_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_points * 2) # Output: 400
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Zurückformen in [Batch, 200, 2]
        x = x.view(-1, self.num_points, 2)
        return x

# Die große Hülle, die beides vereint
class PointNetAutoencoder(nn.Module):
    def __init__(self, num_points=200):
        super().__init__()
        self.encoder = PointNetEncoder2D(global_feat_dim=1024)
        self.decoder = PointNetDecoder2D(global_feat_dim=1024, num_points=num_points)
        
    def forward(self, x):
        fingerabdruck = self.encoder(x)
        rekonstruktion = self.decoder(fingerabdruck)
        return rekonstruktion, fingerabdruck

# # 1. Modell instanziieren
# pointnet = PointNetEncoder2D()

# # WICHTIG: In den Evaluierungs-Modus setzen! 
# # Dadurch "friert" die BatchNorm ein. Ohne das würde die BatchNorm bei 
# # jedem neuen Durchlauf den Mittelwert leicht verändern.
# pointnet.eval()

# # 2. Ein Fake-Muttermal generieren (1 Bild, 200 Punkte, X/Y)
# echtes_muttermal = torch.rand(1, 200, 2)

# # 3. Wir mischen die Punkte komplett durch!
# # torch.randperm(200) gibt uns eine Liste der Zahlen 0-199 in zufälliger Reihenfolge
# misch_index = torch.randperm(200)
# gemischtes_muttermal = echtes_muttermal[:, misch_index, :]

# # 4. Beide durch das PointNet schicken
# with torch.no_grad():
#     fingerabdruck_echt = pointnet(echtes_muttermal)
#     fingerabdruck_gemischt = pointnet(gemischtes_muttermal)

# print("=== POINTNET TEST ===")
# print(f"Shape des echten Inputs:     {echtes_muttermal.shape}")
# print(f"Shape des Output-Vektors:    {fingerabdruck_echt.shape} (Erwartet: [1, 1024])")
# print("-" * 30)

# # 5. Der ultimative Beweis!
# # Wir prüfen, ob die beiden 1024-Vektoren mathematisch absolut identisch sind
# sind_identisch = torch.allclose(fingerabdruck_echt, fingerabdruck_gemischt, atol=1e-6)

# if sind_identisch:
#     print("✅ BOOM! ERFOLG!")
#     print("Obwohl die Punkte komplett durcheinandergemischt wurden,")
#     print("hat PointNet die Geometrie als absolut identisch erkannt.")
# else:
#     print("❌ FEHLER: Die Vektoren sind unterschiedlich.")
    
# # Lass uns die ersten 5 Zahlen des Fingerabdrucks anschauen
# print("-" * 30)
# print(f"Erste 5 Zahlen (Echt):     {fingerabdruck_echt[0, :5].tolist()}")
# print(f"Erste 5 Zahlen (Gemischt): {fingerabdruck_gemischt[0, :5].tolist()}")