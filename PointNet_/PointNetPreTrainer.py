import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class ChamferDistance2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p1, p2):
        """
        p1: Echte Punkte       [Batch, N, 2]
        p2: Rekonstruierte Punkte [Batch, N, 2]
        """

        # Tensoren aufblasen für paarweise Distanzberechnung
        p1_exp = p1.unsqueeze(2)  # [B, N, 1, 2]
        p2_exp = p2.unsqueeze(1)  # [B, 1, N, 2]

        # Paarweise euklidische Distanz
        dist = torch.norm(p1_exp - p2_exp, dim=3)  # [B, N, N]

        # Nächster Nachbar
        min_dist_1_to_2 = torch.min(dist, dim=2)[0]
        min_dist_2_to_1 = torch.min(dist, dim=1)[0]

        # Symmetrische Chamfer Distance
        loss = torch.mean(min_dist_1_to_2) + torch.mean(min_dist_2_to_1)

        return loss


class PointNetPreTrainer:
    def __init__(
        self,
        model,
        dataloader,
        device=None,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=500,
        save_path="best_pointnet_expert_weights.pth"
    ):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.dataloader = dataloader

        self.criterion = ChamferDistance2D()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        self.epochs = epochs
        self.best_loss = float("inf")
        self.save_path = save_path

    def train(self):

        print(f"Starte PointNet Pre-Training auf {self.device}...")

        self.model.train()

        for epoch in range(self.epochs):

            epoch_loss = 0.0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"AE-Epoche {epoch+1}/{self.epochs}"
            )

            for batch_images, batch_points in progress_bar:

                # Bilder werden nicht benötigt
                batch_points = batch_points.to(self.device)

                self.optimizer.zero_grad()

                # Forward
                reconstruction, fingerprint = self.model(batch_points)

                # Loss
                loss = self.criterion(batch_points, reconstruction)

                # Backprop
                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

                progress_bar.set_postfix({
                    "ChamferLoss": f"{loss.item():.5f}"
                })

            avg_loss = epoch_loss / len(self.dataloader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"AE-Epoche {epoch+1} beendet | "
                f"Avg Loss: {avg_loss:.5f} | "
                f"LR: {current_lr}"
            )

            # Bestes Modell speichern
            if avg_loss < self.best_loss:

                self.best_loss = avg_loss

                torch.save(
                    self.model.encoder.state_dict(),
                    self.save_path
                )

                print(
                    f"✅ Neuer Rekord! "
                    f"Encoder gespeichert "
                    f"(Loss: {self.best_loss:.5f})"
                )

            # Scheduler updaten
            self.scheduler.step(avg_loss)

        print("\nTraining beendet!")
        print(
            f"Bester Loss: {self.best_loss:.5f}\n"
            f"Gespeichert unter: {self.save_path}"
        )