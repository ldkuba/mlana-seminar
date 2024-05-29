import torch
import torch.nn.functional as F

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm1d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity # skip connection
        x = self.relu(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, block, input_dim, output_dim):
        super().__init__()
        
        self.input_layer_dim = 128

        self.conv1 = torch.nn.Conv1d(input_dim, self.input_layer_dim, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm1d(self.input_layer_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 128, 3)
        self.layer2 = self._make_layer(block, 192, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 384, 3, stride=2)
        
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(384, output_dim)


    def _make_layer(self, block, input_dim, num_blocks, stride=1):
        downsample = None  
   
        if stride != 1 or input_dim != self.input_layer_dim:
            downsample = torch.nn.Sequential(
                torch.nn.Conv1d(self.input_layer_dim, input_dim, 1, stride, bias=False),
                torch.nn.BatchNorm1d(input_dim),
            )

        layers = []
        layers.append(block(self.input_layer_dim, input_dim, stride, downsample))
        
        self.input_layer_dim = input_dim
        
        for _ in range(1, num_blocks):
            layers.append(block(self.input_layer_dim, input_dim))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet34 = ResNet(ConvolutionBlock, 576, 1152)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet34(x)
        return x
