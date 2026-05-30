import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerBlocks.ImageTransformerBlock import ImageTransformerBlock


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_num_groups=16):
        super().__init__()
        
        self.groupnorm1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

        self.groupnorm2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

        self.resize_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual_connection = x                             
        
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)


        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)                                   

        x = x + self.resize_channels(residual_connection)   
        return x
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
        )

    def forward(self, x):
        return self.upsample(x)
    


class FeatureUNET(nn.Module):
    def __init__(self, in_channels=3, out_features=64, start_dim=64, dim_mults=(1,2,4), residual_blocks_per_group=1, groupnorm_num_groups=16):
        super().__init__()

        self.input_image_channels = in_channels
        channel_sizes = [start_dim * i for i in dim_mults]
        starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]

        self.encoder_config = []

        for idx, d in enumerate(channel_sizes):
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(((d, d), "residual"))

            self.encoder_config.append(((d,d), "downsample"))

            # --- HIER IST DIE RETTUNG FÜR DEN VRAM ---
            # Attention NUR in der tiefsten Schicht (wo das Bild winzig ist) hinzufügen!
            if d == ending_channel_size:
                self.encoder_config.append((d, "attention"))
            # -----------------------------------------

            if idx < len(channel_sizes) - 1:
                self.encoder_config.append(((d, channel_sizes[idx+1]), "residual"))

        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))

        out_dim = ending_channel_size
        reversered_encoder_config = self.encoder_config[::-1]

        self.decoder_config = []
        for idx, (metadata, l_type) in enumerate(reversered_encoder_config):
            
            if l_type != "attention":
                enc_in_channels, enc_out_channels = metadata
                self.decoder_config.append(
                    (
                        (out_dim+enc_out_channels, enc_in_channels), "residual"
                    )
                )

                if l_type == "downsample":
                    self.decoder_config.append(
                        (
                            (enc_in_channels, enc_in_channels), "upsample"
                        )
                    )
                out_dim = enc_in_channels

            else:
                in_channels = metadata
                self.decoder_config.append(
                    (
                        in_channels, "attention"
                    )
                )
        self.decoder_config.append(((starting_channel_size * 2, starting_channel_size), "residual"))

        ### ACTUALLY BUILD MODEL ###
        self.conv_in_proj = nn.Conv2d(self.input_image_channels, starting_channel_size, kernel_size=3, padding="same")

        self.encoder = nn.ModuleList()
        for metadata, l_type in self.encoder_config:
            if l_type == "residual":
                in_channels, out_channels = metadata
                # Zeit-Parameter entfernt!
                self.encoder.append(ResidualBlock(in_channels, out_channels, groupnorm_num_groups))
            elif l_type == "downsample":
                in_channels, out_channels = metadata
                self.encoder.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            elif l_type == "attention":
                in_channels = metadata
                self.encoder.append(ImageTransformerBlock(in_channels))

        self.bottleneck = nn.ModuleList()
        for (in_channels, out_channels), _ in self.bottleneck_config:
            # Zeit-Parameter entfernt!
            self.bottleneck.append(ResidualBlock(in_channels, out_channels, groupnorm_num_groups))

        self.decoder = nn.ModuleList()
        for metadata, l_type in self.decoder_config:
            if l_type == "residual":
                in_channels, out_channels = metadata
                # Zeit-Parameter entfernt!
                self.decoder.append(ResidualBlock(in_channels, out_channels, groupnorm_num_groups))
            elif l_type == "upsample":
                in_channels, out_channels = metadata
                self.decoder.append(UpsampleBlock(in_channels, out_channels))
            elif l_type == "attention":
                in_channels = metadata
                self.decoder.append(ImageTransformerBlock(in_channels))

        # NEU: Wir spucken nicht input_image_channels (3) aus, sondern out_features (64)!
        self.conv_out_proj = nn.Conv2d(starting_channel_size, out_features, kernel_size=3, padding="same")

    def forward(self, x): 
        residuals = []
        multi_scale_features = []

        x = self.conv_in_proj(x)
        residuals.append(x)  

        # --- ENCODER ---
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x)
                residuals.append(x)
            elif isinstance(module, nn.Conv2d):  
                x = module(x)
                residuals.append(x)
            else:  
                x = module(x)

        # --- BOTTLENECK ---
        for module in self.bottleneck:
            x = module(x) 
            
        # HINWEIS: Hier im Keller speichern wir NICHTS mehr ab! 
        # Das Signal geht erst in den Decoder, um dein "Kandidat 2" zu werden.

        # --- DECODER ---
        upsample_count = 0
        for module in self.decoder:
            
            # WENN EIN AUFZUG KOMMT:
            if isinstance(module, UpsampleBlock):
                upsample_count += 1
                
                # 🎯 MESSFÜHLER 1 (Grob - Dein Kandidat 2)
                # Das Signal ist 72x96 und hat die Skip-Connection aus dem Keller verdaut.
                # Wir speichern es ab, BEVOR es vergrößert wird!
                if upsample_count == 1:
                    multi_scale_features.append(x)
                    
                # 🎯 MESSFÜHLER 2 (Mittel - Dein Kandidat 3)
                # Das Signal ist 144x192 und hat die zweite Skip-Connection verdaut.
                elif upsample_count == 2:
                    multi_scale_features.append(x)
                    
                # Jetzt darf das Modul das Bild vergrößern
                x = module(x)
                
            # WENN EIN NORMALER BLOCK KOMMT:
            elif isinstance(module, ResidualBlock):
                residual_tensor = residuals.pop()
                
                if x.shape[2:] != residual_tensor.shape[2:]:
                    x = F.interpolate(x, size=residual_tensor.shape[2:], mode="bilinear", align_corners=False)
                
                x = torch.cat([x, residual_tensor], dim=1)
                x = module(x) 
            
            # WENN ATTENTION KOMMT:
            else:
                x = module(x)

        # --- OUTPUT ---
        x = self.conv_out_proj(x)
        
        # 🎯 MESSFÜHLER 3 (Fein - Dein Kandidat 4)
        # Volle 576x768 Pixel. Die absolute Endausgabe des Modells.
        multi_scale_features.append(x)
        
        return multi_scale_features