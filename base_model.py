import torch
import torchvision

class Transformer(torch.nn.Module):
    def __init__(self, model='vit', freeze_layers=False):
        '''
        :param model: vit, swin, resnet110 or resnet50
        :param freeze_layers: if True - freeze some layers
        '''
        super().__init__()
        if model == 'vit':
            model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            # model.heads = torch.nn.Linear(1024, 2)  # можно заморозить некоторые слои, надо экспериментировать
        elif model == 'swin':
            model = torchvision.models.swin_v2_b(weights='IMAGENET1K_V1') # трансформер полегче в 3 раза
            # model.head = torch.nn.Linear(1024, 2)
        self.linear = torch.nn.Linear(1024, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.model(x)  # out for ensemble
        out2 = self.linear(self.relu(out1))
        return out1, out2