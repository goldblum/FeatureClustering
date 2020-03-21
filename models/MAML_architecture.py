import torch.nn as nn
import math

class MAML_net(nn.Module):
    def __init__(self, input_channels=3, num_filters=32, num_classes=5, bias=True):
        super(MAML_net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.head = nn.Linear(5*5*num_filters, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.head(x.view(x.size(0), -1))
        return x

class MAML_Embedding(nn.Module):
    def __init__(self, input_channels=3, num_filters=32, bias=True):
        super(MAML_Embedding, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))
        return x.view(x.size(0), -1)
