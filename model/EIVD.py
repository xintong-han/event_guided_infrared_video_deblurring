import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import conv1x1, conv3x3, conv5x5, actFunc


# 动态生长率的密集层
class dynamic_dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dynamic_dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)
        # 动态缩放因子
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        out = self.act(self.conv(x))
        out = out * self.scale  # 应用动态缩放
        out = torch.cat((x, out), 1)
        return out


# 动态生长率残差密集块 (Dynamic Growth RDB)
class DynamicGrowthRDB(nn.Module):
    def __init__(self, in_channels, base_growth, num_layer, activation='relu'):
        super(DynamicGrowthRDB, self).__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_layer):
            self.layers.append(dynamic_dense_layer(
                current_channels,
                base_growth,
                activation
            ))
            current_channels += base_growth

        self.conv1x1 = conv1x1(current_channels, in_channels)
        # 模糊程度感知器，用于调整整体缩放
        self.blur_perceiver = nn.Sequential(
            conv1x1(in_channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 感知模糊程度并生成缩放因子
        blur_scale = self.blur_perceiver(x)[:, :, None, None]
        out = x

        for layer in self.layers:
            out = layer(out)

        out = self.conv1x1(out)
        # 根据模糊程度动态调整残差连接强度
        out = x + blur_scale * out
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            # 使用动态生长率RDB替代原有RDB
            self.RDBs.append(DynamicGrowthRDB(
                in_channels,
                growthRate,
                num_layer,
                activation
            ))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out


# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        # 使用动态生长率RDB
        self.rdb = DynamicGrowthRDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        x = self.rdb(x)
        out = self.down_sampling(x)
        return out


# 事件特征提取器，加入时间编码
class EventFeatureExtractor(nn.Module):
    def __init__(self, para):
        super(EventFeatureExtractor, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        # 事件输入: 2通道 (正负事件)
        self.F_B0 = conv5x5(2, self.n_feats, stride=1)
        # 时间特征编码层
        self.time_encoder = conv1x1(1, self.n_feats)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)

    def forward(self, x, event_timestamps):
        out = self.F_B0(x)
        # 编码并融合时间信息
        time_feat = self.time_encoder(event_timestamps)
        out = out + time_feat  # 注入时间信息
        out = self.F_B1(out)
        out = self.F_B2(out)
        return out


# 图像特征提取器
class ImageFeatureExtractor(nn.Module):
    def __init__(self, para):
        super(ImageFeatureExtractor, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        # 图像输入: 3通道 (RGB)
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)

        # 模糊区域检测器，用于引导注意力
        self.blur_detector = nn.Sequential(
            conv3x3(3, self.n_feats // 2),
            actFunc(self.activation),
            conv3x3(self.n_feats // 2, 1),
            nn.Sigmoid()  # 输出模糊程度图 (0-1)
        )

        # 用于模糊图的1x1卷积层
        self.blur_conv = conv1x1(1, self.n_feats)

    def forward(self, x):
        # 检测模糊区域
        blur_map = self.blur_detector(x)
        out = self.F_B0(x)
        # 将模糊信息融入特征
        out = out + self.blur_conv(blur_map)
        out = self.F_B1(out)
        out = self.F_B2(out)
        return out, blur_map


# 动态注意力融合模块（名称修改：DDGAFusion）
class DDGAFusion(nn.Module):
    def __init__(self, para):
        super(DDGAFusion, self).__init__()
        self.n_feats = para.n_features
        # 图像引导的事件注意力（图像模糊区域增强事件权重）
        self.img2evt_attn = nn.Sequential(
            conv1x1(4 * self.n_feats + 1, 2 * self.n_feats),  # +1 加入模糊图
            actFunc(para.activation),
            conv1x1(2 * self.n_feats, 4 * self.n_feats),
            nn.Sigmoid()
        )
        # 事件引导的图像注意力（事件密集区域增强图像权重）
        self.evt2img_attn = nn.Sequential(
            conv1x1(4 * self.n_feats, 2 * self.n_feats),
            actFunc(para.activation),
            conv1x1(2 * self.n_feats, 4 * self.n_feats),
            nn.Sigmoid()
        )
        # 融合后处理
        self.fusion_conv = conv3x3(4 * self.n_feats, 4 * self.n_feats)

    def forward(self, img_feat, event_feat, blur_map):
        # 上采样模糊图以匹配特征尺寸
        blur_map = F.interpolate(blur_map, size=img_feat.shape[2:], mode='bilinear', align_corners=True)

        # 图像特征引导事件特征的注意力
        img_feat_with_blur = torch.cat([img_feat, blur_map], dim=1)  # 加入模糊信息
        evt_attn = self.img2evt_attn(img_feat_with_blur)  # 基于图像内容和模糊程度生成事件注意力图
        weighted_evt = event_feat * evt_attn

        # 事件特征引导图像特征的注意力
        img_attn = self.evt2img_attn(event_feat)  # 基于事件生成图像注意力图
        weighted_img = img_feat * img_attn

        # 融合并增强
        fused = weighted_img + weighted_evt
        return self.fusion_conv(fused)


# 门控隐藏状态更新模块
class GatedHiddenUpdate(nn.Module):
    def __init__(self, para):
        super(GatedHiddenUpdate, self).__init__()
        self.n_feats = para.n_features
        # 输入：融合特征+隐藏状态 (5n_feats)
        self.update_gate = nn.Sequential(
            conv3x3(5 * self.n_feats, self.n_feats),
            nn.Sigmoid()  # 0-1控制更新强度
        )
        self.reset_gate = nn.Sequential(
            conv3x3(5 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )
        self.new_state = nn.Sequential(
            conv3x3(5 * self.n_feats + self.n_feats, self.n_feats),  # 加入重置后的隐藏状态
            DynamicGrowthRDB(in_channels=self.n_feats, base_growth=self.n_feats, num_layer=3),
            nn.Tanh()
        )

    def forward(self, x, s_prev):  # x: 融合特征+历史隐藏态, s_prev: 上一时刻隐藏态
        z = self.update_gate(x)  # 更新门
        r = self.reset_gate(x)  # 重置门
        # 结合重置后的隐藏状态计算新状态
        s_new = self.new_state(torch.cat([x, r * s_prev], dim=1))
        # 动态更新隐藏态
        s = (1 - z) * s_prev + z * s_new
        return s


# 改进的RDB-based RNN cell with event fusion（名称修改：BGRCell）
class BGR(nn.Module):
    def __init__(self, para):
        super(BGR, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # RDNet处理融合特征与隐藏状态
        self.F_R = RDNet(in_channels=5 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation).to(self.device)

        # 隐藏状态处理 - 使用门控更新替代原有处理
        self.hidden_update = GatedHiddenUpdate(para).to(self.device)

        self.to(self.device)

    def forward(self, fused_feat, s_last):
        # 确保输入都在正确设备上
        fused_feat = fused_feat.to(self.device)
        s_last = s_last.to(self.device)

        # 与隐藏状态结合
        out = torch.cat([fused_feat, s_last], dim=1)

        # 用RDNet处理
        out = self.F_R(out)

        # 门控更新隐藏状态
        s = self.hidden_update(out, s_last)

        return out, s


# Global spatio-temporal attention module
class GSA(nn.Module):
    def __init__(self, para):
        super(GSA, self).__init__()
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.F_f = nn.Sequential(
            nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            actFunc(para.activation),
            nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
            nn.Sigmoid()
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        # condense layer
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        # fusion layer
        self.fusion = conv1x1(self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))

    def forward(self, hs):
        # hs: [(n=4,c=80,h=64,w=64), ..., (n,c,h,w)]
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
                if len(w.shape) == 1:
                    w = w.unsqueeze(dim=0)
                w = self.F_f(w)
                w = w.reshape(*w.shape, 1, 1)
                cor = self.F_p(cor)
                cor = self.condense(w * cor)
                cor_l.append(cor)
        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))
        return out


# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    """
    改进的高效时空循环神经网络与事件融合 (ESTRNN-Event)
    加入了动态注意力融合、门控隐藏状态更新和双向时序处理
    """

    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化特征提取器和融合模块（使用修改后的DDGAFusion）
        self.image_extractor = ImageFeatureExtractor(para).to(self.device)
        self.event_extractor = EventFeatureExtractor(para).to(self.device)
        self.attention_fusion = DDGAFusion(para).to(self.device)
        
        # 初始化RNN单元（使用修改后的BGRCell）
        self.cell = BGR(para).to(self.device)
        
        # 初始化其他模块
        self.recons = Reconstructor(para).to(self.device)
        self.fusion = GSA(para).to(self.device)

        # 将整个模型移动到指定设备
        self.to(self.device)

    def forward(self, x, profile_flag=False):
        # 确保输入数据在正确设备上
        x = [item.to(self.device) for item in x]

        # 确保输入是4个元素
        if len(x) != 4:
            raise ValueError(f"模型期望4个输入，实际收到{len(x)}个")

        blur_imgs, sharp_imgs, event_maps, event_timestamps = x

        # 验证时间戳维度是否正确 (B, T, 1, H, W)
        if len(event_timestamps.shape) != 5:
            # 如果是4维 (B, T, H, W)，添加通道维度
            if len(event_timestamps.shape) == 4:
                event_timestamps = event_timestamps.unsqueeze(2)
            else:
                raise ValueError(f"事件时间戳维度不正确，期望5维，实际{len(event_timestamps.shape)}维")

        if profile_flag:
            return self.profile_forward(x)

        outputs, forward_hs, backward_hs = [], [], []

        # 检查并修正 blur_imgs 的维度
        if len(blur_imgs.shape) > 5:
            # 移除多余的维度
            blur_imgs = torch.squeeze(blur_imgs, dim=2)
            print(f"修正后 blur_imgs 维度: {blur_imgs.shape}")
        elif len(blur_imgs.shape) < 5:
            raise ValueError(f"blur_imgs 维度不足，期望5维，实际{len(blur_imgs.shape)}维")
        batch_size, frames, channels, height, width = blur_imgs.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)

        # 正向时序处理
        s_forward = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            # 提取图像和事件特征
            img_feat, blur_map = self.image_extractor(blur_imgs[:, i, :, :, :])
            event_feat = self.event_extractor(event_maps[:, i, :, :, :], event_timestamps[:, i, :, :, :])
            
            # 融合特征
            fused_feat = self.attention_fusion(img_feat, event_feat, blur_map)
            
            # RNN单元接收融合后的特征和隐藏状态
            h, s_forward = self.cell(fused_feat, s_forward)
            forward_hs.append(h)

        # 反向时序处理
        s_backward = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in reversed(range(frames)):
            # 提取图像和事件特征
            img_feat, blur_map = self.image_extractor(blur_imgs[:, i, :, :, :])
            event_feat = self.event_extractor(event_maps[:, i, :, :, :], event_timestamps[:, i, :, :, :])
            
            # 融合特征
            fused_feat = self.attention_fusion(img_feat, event_feat, blur_map)
            
            # RNN单元接收融合后的特征和隐藏状态
            h, s_backward = self.cell(fused_feat, s_backward)
            backward_hs.append(h)
        backward_hs = backward_hs[::-1]  # 反转回原顺序

        # 融合双向特征
        hs = [forward_hs[i] + backward_hs[i] for i in range(frames)]

        # 生成目标帧输出
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)

    # 用于计算GMACs
    def profile_forward(self, x):
        blur_imgs, sharp_imgs, event_maps, event_timestamps = x
        outputs, hs = [], []
        batch_size, frames, channels, height, width = blur_imgs.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)

        for i in range(frames):
            # 提取图像和事件特征
            img_feat, blur_map = self.image_extractor(blur_imgs[:, i, :, :, :])
            event_feat = self.event_extractor(event_maps[:, i, :, :, :], event_timestamps[:, i, :, :, :])
            
            # 融合特征
            fused_feat = self.attention_fusion(img_feat, event_feat, blur_map)
            
            # RNN单元处理
            h, s = self.cell(fused_feat, s)
            hs.append(h)

        for i in range(self.num_fb + self.num_ff):
            hs.append(torch.randn(*h.shape).to(self.device))

        for i in range(self.num_fb, frames + self.num_fb):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)


def feed(model, iter_samples):
    # 检查输入数据维度
    blur_imgs = iter_samples[0]
    if len(blur_imgs.shape) != 5:
        raise ValueError(f"feed 函数中 blur_imgs 维度错误，期望5维，实际{len(blur_imgs.shape)}维")
    outputs = model(iter_samples)
    return outputs


def cost_profile(model, H, W, seq_length):
    # 获取模型所在设备
    device = next(model.parameters()).device

    # 创建用于性能分析的虚拟数据，并确保在同一设备上
    blur_imgs = torch.randn(1, seq_length, 3, H, W).to(device)
    sharp_imgs = torch.randn(1, seq_length, 3, H, W).to(device)
    event_maps = torch.randn(1, seq_length, 2, H, W).to(device)
    event_timestamps = torch.randn(1, seq_length, 1, H, W).to(device)  # 时间戳数据

    x = [blur_imgs, sharp_imgs, event_maps, event_timestamps]
    profile_flag = True
    # 确保模型在正确设备上
    model = model.to(device)
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params
    