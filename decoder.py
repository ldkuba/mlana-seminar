import torch
import torch.nn.functional as F

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm1d(out_channels)

        self.downsample = torch.nn.Sequential()
        if in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)

        identity = self.downsample(identity)
        x += identity # skip connection
        x = self.relu(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, block, input_dim, output_dim):
        super().__init__()
        
        self.input_layer_dim = 128

        self.conv1 = torch.nn.Conv1d(input_dim, self.input_layer_dim, kernel_size=7, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm1d(self.input_layer_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.layer_norm = torch.nn.LayerNorm(self.input_layer_dim)
        
        self.layer1 = self._make_layer(block, 128, 3)
        self.layer2 = self._make_layer(block, 192, 4)
        self.layer3 = self._make_layer(block, 256, 6)
        self.layer4 = self._make_layer(block, 384, 3)
        
        self.fc = torch.nn.Linear(384, output_dim)


    def _make_layer(self, block, input_dim, num_blocks):
        layers = []
        layers.append(block(self.input_layer_dim, input_dim))
        
        self.input_layer_dim = input_dim
        
        for _ in range(1, num_blocks):
            layers.append(block(self.input_layer_dim, input_dim))

        return torch.nn.Sequential(*layers)

    #: x: (batch_size, num_faces, face_feature_dim)
    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = torch.swapaxes(x, 1, 2)
        x = self.layer_norm(x)
        x = torch.swapaxes(x, 1, 2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.swapaxes(x, 1, 2)
        x = self.fc(x)

        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet34 = ResNet(ConvolutionBlock, 576, 1152)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet34(x)
        return x
