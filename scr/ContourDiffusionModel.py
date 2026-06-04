import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContourDiffusionModel(nn.Module):
    def __init__(self, n_punkte=200, hidden_dim=128, num_layers=4):
        super().__init__()
        self.n_punkte = n_punkte
        
        # 1. Zeit-Einbettung
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.GELU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Koordinaten-Einbettung
        self.coord_mlp = nn.Linear(2, hidden_dim)
        
        # 3. Zeit-gesteuerter Feature-Mixer
        self.multi_scale_dim = 64
        
        # --- FIX: DAS FUSION MLP WIRD GRÖßER ---
        # Es empfängt jetzt: Lokale Features (448) + Globale Features (448) + Zeit (128)
        self.fusion_mlp = nn.Sequential(
            nn.Linear((self.multi_scale_dim * 2) + hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # --- Sinus-Embedding (Dein fixes Koordinatensystem) ---
        pe = torch.zeros(1, n_punkte, hidden_dim)
        position = torch.arange(0, n_punkte, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('seq_emb', pe)
        
        # --- Das Bügeleisen (Lokale Glättung) ---
        self.local_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode='circular')
        
        # 4. Der globale Point-Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.local_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode='circular')
        
        # --- Skip Connections ---
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 2, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, punkte, zeit, wetterkarten_liste):
        batch_size = punkte.shape[0]
        
        # 1. Zeit berechnen
        t = zeit.view(-1, 1).float() / 1000.0 
        t_emb = self.time_mlp(t).unsqueeze(1) 
        
        # --- 2. FEATURES EXTRAHIEREN (LOKAL & GLOBAL) ---
        grid_coords = torch.clamp(punkte, min=-1.0, max=1.0)
        grid = grid_coords.unsqueeze(1) 
        
        gesammelte_features_lokal = []
        gesammelte_features_global = [] # NEU
        
        for karte in wetterkarten_liste:
            # A. Lokale Nadelstiche (Wie vorher)
            sampled = F.grid_sample(karte, grid, mode='bilinear', padding_mode='border', align_corners=True)
            sampled = sampled.squeeze(2).transpose(1, 2) 
            gesammelte_features_lokal.append(sampled)
            
            # B. Das NEUE Globale Radar (Durchschnitt über die gesamte Karte)
            # karte hat Shape [Batch, Channels, H, W] -> mean(dim=[2, 3]) bricht es auf [Batch, Channels] herunter
            pooled = karte.mean(dim=[2, 3]) 
            gesammelte_features_global.append(pooled)
            
        all_features_local = torch.cat(gesammelte_features_lokal, dim=-1) # Shape: [Batch, 200, 448]
        global_radar = torch.cat(gesammelte_features_global, dim=-1)      # Shape: [Batch, 448]
        
        # Das Radar für alle 200 Punkte kopieren (damit jeder Punkt das Gesamtbild kennt)
        global_radar_expanded = global_radar.unsqueeze(1).expand(-1, self.n_punkte, -1) # Shape: [Batch, 200, 448]
        
        # --- 3. ALLES FUSIONIEREN ---
        t_emb_expanded = t_emb.expand(-1, self.n_punkte, -1) 
        
        # Wir füttern das MLP jetzt mit Lokal + Global + Zeit
        fusion_input = torch.cat([all_features_local, global_radar_expanded, t_emb_expanded], dim=-1) 
        x_features = self.fusion_mlp(fusion_input) 
        
        # 4. Einbettungen addieren
        x_coords = self.coord_mlp(punkte)            
        x = x_coords + x_features + self.seq_emb + t_emb
        
        # --- DER SIGNALWEG ---
        x = x.transpose(1, 2)
        x = F.gelu(self.local_conv1(x))
        x = x.transpose(1, 2) 
        
        x = self.transformer(x)
        
        x = x.transpose(1, 2)
        x = F.gelu(self.local_conv2(x))
        x = x.transpose(1, 2)
        
        out_input = torch.cat([x, x_features, punkte], dim=-1)
        predicted_noise = self.output_mlp(out_input)
        
        return predicted_noise