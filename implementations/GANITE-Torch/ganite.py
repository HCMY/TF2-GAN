import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, feature_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim,2*feature_dim, bias=True)
        self.fc2 = torch.nn.Linear(2*feature_dim, 3*feature_dim, bias=True)
        self.fc3 = torch.nn.Linear(3*feature_dim, feature_dim, bias=True)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return x

class CFDiscriminator(torch.nn.Module):
    def __init__(self, feature_dim, n_treatment):
        """
        :param feature_dim:
        """
        super(CFDiscriminator, self).__init__()
        
        self.mlp = MLP(feature_dim=feature_dim)
        self.fc = torch.nn.Linear(feature_dim, n_treatment, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, t, yf, y_wave):
        inp0 = (1. - t)*yf + t*y_wave[:, 0].view(-1,1)
        inp1 = t * yf + (1. - t)*y_wave[:,1].view(-1,1)
        y_bar = torch.cat((inp0, inp1),dim=1)
        inp = torch.cat((x, y_bar), dim=1)

        out = self.mlp(inp)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


class CFGenerator(torch.nn.Module):
    def __init__(self,feature_dim, n_treatment):
        """
        :param feature_dim:
        """
        super(CFGenerator, self).__init__()
        self.mlp = MLP(feature_dim=feature_dim)
        self.fc_model_group = torch.nn.Linear(feature_dim,  n_treatment, bias=True)
        self.fc_control_group = torch.nn.Linear(feature_dim, n_treatment, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, t, yf, z_g):
        inp = torch.cat((x, t, yf, z_g), dim=1)

        out = self.mlp(inp)
        factual_outcome = self.fc_model_group(out)
        count_factual_outcome = self.fc_control_group(out)
        outcomes = torch.cat((factual_outcome, count_factual_outcome),dim=1)
        outcomes = self.sigmoid(outcomes)

        return outcomes


class InferenceNet(torch.nn.Module):
    def __init__(self,feature_dim, n_treatment):
        super(InferenceNet,self).__init__()
        self.mlp = MLP(feature_dim=feature_dim)
        self.fc_outcomes = torch.nn.Linear(feature_dim,  n_treatment*2, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.fc_outcomes(x)
        outcomes = self.sigmoid(x)

        return outcomes




