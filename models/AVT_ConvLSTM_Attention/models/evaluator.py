import torch.nn as nn


class MLP_block(nn.Module):

    def __init__(self, feature_dim, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        output = self.softmax(self.layer4(x))
        return output


class Evaluator(nn.Module):

    def __init__(self, feature_dim, output_dim, predict_type, num_subscores=None):
        super(Evaluator, self).__init__()
        assert predict_type in ['phq-subscores', 'phq-score', 'phq-binary'], \
            "Argument --predict_type in config['MODEL']['EVALUATOR'] could only be ['phq-subscores', 'phq-score', 'phq-binary']"

        self.predict_type = predict_type

        if self.predict_type == 'phq-subscores':
            # use multi-head model to predict the subscores
            assert num_subscores is not None, 'num_subscores is required in multi-head model'
            self.evaluator = nn.ModuleList([MLP_block(feature_dim, output_dim) for _ in range(num_subscores)])

        else:
            # use single-head model to predict the PHQ Score or Binary Depression Classification
            self.evaluator = MLP_block(feature_dim, output_dim)

    def forward(self, feats_avg):  # data: NCTHW
        if self.predict_type == 'phq-subscores':
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_subscores
        
        else:
            probs = self.evaluator(feats_avg)  # Nxoutput_dim
        
        return probs


if __name__ == '__main__':

    output_feature_dim = 1024
    n_classes = 4
    n_subscores = 8

    evaluator = Evaluator(output_feature_dim,
                          n_classes,
                          predict_type='phq-subscores',
                          num_subscores=n_subscores)
    # print(evaluator.__dict__)
    print(*evaluator.parameters())
    print('done!')