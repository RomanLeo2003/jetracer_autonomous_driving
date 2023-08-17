import torch
import torchvision

class Ensemble(torch.nn.Module):
    def __init__(self, models: list, in_features=1024, ensemble_strategy='stacking'):
        '''
        :param models: list of models
        :param in_features: dim of in features
        :param ensemble_strategy: stacking, bagging or voting
        '''
        super().__init__()
        self.ensemble_strategy = ensemble_strategy
        self.models = [self.freeze_model(model) for model in models]
        self.linear = torch.nn.Linear(in_features, 2)
        self.leaky_relu = torch.nn.LeakyReLU()

    def freeze_model(self, model):
        # todo: remove last layer
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, x):
        out = torch.Tensor([])
        if self.ensemble_strategy == 'stacking':
            x = torch.cat([model(x)[1] for model in self.models])
            out = self.linear(self.leaky_relu(x))
        elif self.ensemble_strategy == 'bagging':
            out = sum([model(x) for model in self.models]) / len(self.models)
        elif self.ensemble_strategy == 'voting':
            pass
        return out
