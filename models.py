import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TAM(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(TAM, self).__init__()
        self.h = h
        self.w = w


        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))


        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()


        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=7, padding=3)
        )
        self.spatial_sigmoid = nn.Sigmoid()

        self.pixel_attention = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )


        self.highlight_detector = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.lowlight_detector = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        channel_attention = x * s_h.expand_as(x) * s_w.expand_as(x)


        avg_out = torch.mean(channel_attention, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_attention, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_sigmoid(self.spatial_conv(spatial_features))


        grayscale = torch.mean(x, dim=1, keepdim=True)


        mean_value = torch.mean(grayscale)
        highlight_mask = (grayscale > mean_value).float()
        highlight_attention = self.highlight_detector(grayscale * highlight_mask)


        lowlight_mask = (grayscale <= mean_value).float()
        lowlight_attention = self.lowlight_detector(grayscale * lowlight_mask)


        combined_pixel_attention = torch.cat([
            grayscale,
            highlight_attention,
            lowlight_attention
        ], dim=1)

        pixel_attention = self.pixel_attention(combined_pixel_attention)


        out = channel_attention * spatial_attention * pixel_attention

        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        x = x.transpose(0, 1)  # (B, N, D) -> (N, B, D)
        x_ln = self.norm1(x)
        attention_output, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        x = x.transpose(0, 1)  # (N, B, D) -> (B, N, D)
        return x





import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock





class RSCNN_Parallel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
       
        self.resnet = resnet50(weights='IMAGENET1K_V1')
        
    
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, 
                                     padding=3, bias=False)
      
        with torch.no_grad():
            self.resnet.conv1.weight[:,0] = torch.mean(old_conv.weight, dim=1)
        
      
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.ca_resnet = CA_Block(2048, 7, 7)
        
     
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', 
                                    pretrained=False,
                                    in_chans=1,
                                    num_classes=0)
        self.ca_swin = CA_Block(768, 7, 7)
        
      
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048+768, num_classes)
       
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out')
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
       
        res_feat = self.resnet(x)
        res_feat = self.ca_resnet(res_feat)
        res_feat = self.avgpool(res_feat).flatten(1)
        
   
        swin_feat = self.swin.forward_features(x)
        swin_feat = swin_feat.transpose(1,2).reshape(-1,768,7,7)
        swin_feat = self.ca_swin(swin_feat)
        swin_feat = self.avgpool(swin_feat).flatten(1)
        
        return self.fc(torch.cat([res_feat, swin_feat], dim=1))



