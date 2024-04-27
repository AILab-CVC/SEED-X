import torch.nn as nn
import pyrootutils


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

class DiscreteModleIdentity(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Identity()

    def forward(self, image_embeds, input_ids=None, text_attention_mask=None, text_embeds=None):
        return

    def encode_image_embeds(self, image_embeds):
        return self.model(image_embeds)
