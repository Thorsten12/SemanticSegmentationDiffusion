import sys
sys.path.insert(0, "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained")
import torch
from dinov3.models.vision_transformer import vit_small

model = vit_small(patch_size=16, n_storage_tokens=4)
sd = torch.load(
    "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    map_location="cpu",
)
if "model" in sd:
    sd = sd["model"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)

model.eval()
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    feats = model.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True, return_class_token=False, norm=True)
for i, f in enumerate(feats):
    print(f"Layer {[2,5,8,11][i]}: shape={tuple(f.shape)}")