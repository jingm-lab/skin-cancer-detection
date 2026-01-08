# %%
import torch.nn as nn
import torch.nn.functional as F


# %%
class SkinCancerCNN(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        # self.n_features = n_features
        self.p = p

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 32 * 128 * 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 * 64 * 64

            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 64 * 64 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 * 32 * 32

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding =1), # 128 * 32 * 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride = 2), # 128 * 16 * 16
            # nn.AdaptiveAvgPool2d((1,1)), # 128 *1 * 1
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            # nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
            nn.Linear(1024, 1),
            # nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
