#https://blog.csdn.net/rainforestgreen/article/details/85157989
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
 
    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))
 
class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)
 
    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x
 
class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
    def forward(self, input):
        # global average pooling
        x = torch.mean(input, 3, keepdim=True)
        x = torch.mean(x, 2, keepdim=True)
        assert self.in_channels == x.size(1), 'in_channels {} and out_channels {} should all be {}'.format(self.in_channels,x.size(1),x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x
 
 
class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes,in_channels=1024):
        super().__init__()
        self.in_channels = in_channels  
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels {} of ConvBlock should be {}'.format(self.in_channels,x.size(1))
        feature = self.convblock(x)
        x = torch.mean(feature, 3, keepdim=True)
        x = torch.mean(x, 2 ,keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.relu(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x
 
class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()
 
        # build context path
        self.context_path = build_contextpath(name=context_path)  #这里其实就是特征提取的基本网络，主要用到了res18和res101
		
        # build attention refinement module  
        if context_path=='resnet18':
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        elif context_path=='resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
        else:
            raise 'context_path error'
 
        # build feature fusion module
        if context_path=='resnet18':
            self.feature_fusion_module = FeatureFusionModule(num_classes,1024) #此处源码没有实现，因此会有错误。我进行了分析和实现
        elif context_path=='resnet101':
            self.feature_fusion_module = FeatureFusionModule(num_classes,3328)
        else:
            raise 'context_path error'
 
        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
 
    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)
 
        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)
 
        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)
 
        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)
        return result
