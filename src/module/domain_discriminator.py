from torch import nn


class DomainDiscriminator(nn.Sequential):
    """
    Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/domain_discriminator.py)
    """
    def __init__(self, in_feature, hidden_size, sigmoid=True):
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)
            
        super(DomainDiscriminator, self).__init__(
            nn.Linear(in_feature, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            final_layer
        )

    def get_parameters(self):
        return [{"params": self.parameters(), "lr": 1.}]