import torch.nn as nn
from .fusion import Bottleneck

class MLP_block(nn.Module):

    def __init__(self, feature_dim, output_dim, dropout, attention_config):
        super(MLP_block, self).__init__()
        self.feature_dim = feature_dim
        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.layer4 = nn.Linear(64, output_dim)
        self.drop = nn.Dropout(dropout)
        self.attention = Bottleneck(inplanes=attention_config['INPUT_DIM'], 
                                    planes=attention_config['HIDDEN_DIM'],
                                    base_width=attention_config['BASE_WIDTH'],
                                    fuse_type=attention_config['FUSE_TYPE'])

    def forward(self, x):
        B, C, H, W = x.shape

        assert self.feature_dim == H*W, \
            f"Argument --INPUT_FEATURE_DIM in config['MODEL']['EVALUATOR'] should be equal to {H*W} (num_modal x feature_dim of each branch))"

        x = self.attention(x).view(B, -1)

        x = self.activation(self.bn1(self.layer1(x)))
        x = self.activation(self.bn2(self.layer2(x)))
        x = self.activation(self.bn3(self.layer3(x)))
        output = self.softmax(self.layer4(x))

        return output


class Evaluator(nn.Module):

    def __init__(self, feature_dim, output_dim, predict_type, dropout, attention_config, num_subscores=None):
        super(Evaluator, self).__init__()
        assert predict_type in ['phq-subscores', 'phq-score', 'phq-binary'], \
            "Argument --predict_type in config['MODEL']['EVALUATOR'] could only be ['phq-subscores', 'phq-score', 'phq-binary']"

        self.predict_type = predict_type

        if self.predict_type == 'phq-subscores':
            # use multi-head model to predict the subscores
            assert num_subscores is not None, 'num_subscores is required in multi-head model'
            self.evaluator = nn.ModuleList([MLP_block(feature_dim, output_dim, dropout, attention_config) for _ in range(num_subscores)])

        else:
            # use single-head model to predict the PHQ Score or Binary Depression Classification
            self.evaluator = MLP_block(feature_dim, output_dim, attention_config)

    def forward(self, feats_avg):
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