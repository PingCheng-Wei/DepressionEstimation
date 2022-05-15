import torch.nn as nn


class MLP_block(nn.Module):

    def __init__(self, feature_dim, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        return output


class Evaluator(nn.Module):

    def __init__(self, feature_dim, output_dim, num_subscores=None):
        super(Evaluator, self).__init__()

        self.model_type = 'MUSDL'

        assert num_subscores is not None, 'num_subscores is required in MUSDL'
        self.evaluator = nn.ModuleList([MLP_block(feature_dim, output_dim) for _ in range(num_subscores)])

    def forward(self, feats_avg):  # data: NCTHW
        probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_subscores
        return probs


if __name__ == '__main__':

    output_feature_dim = 1024
    n_classes = 4
    n_subscores = 8

    evaluator = Evaluator(output_feature_dim,
                          n_classes,
                          n_subscores )
    # print(evaluator.__dict__)
    print(*evaluator.parameters())
    print('done!')