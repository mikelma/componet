import torch
import torchvision.transforms as T


class DinoEncoder:
    def __init__(self, dino_size, device):
        # see: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md#model-details
        if dino_size == "s":
            self.embed_dim = 384
        elif dino_size == "b":
            self.embed_dim = 768
        elif dino_size == "l":
            self.embed_dim = 1024
        elif dino_size == "g":
            self.embed_dim = 1536
        else:
            print(f"Invalid DINOv2 size {dino_size}. Valid values are s, b, l, and g")

        self.dino = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_vit{dino_size}14_reg"
        ).to(device)
        self.transform = T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    @torch.no_grad()
    def encode(self, x):
        # x: (batch, n_stack, h, w, n_channels)
        batch_size = x.size(0)
        n_stack = x.size(1)

        x = torch.flatten(
            x, start_dim=0, end_dim=1
        )  # x: (batch * n_stack, h, w, n_channels)
        x = x.permute(0, 3, 1, 2)  # x: (batch * n_stack, n_channles, h, w)

        x = self.transform(x)  # normalize
        latent = self.dino(x)
        latent = latent.resize(batch_size, self.embed_dim * n_stack)

        return latent
