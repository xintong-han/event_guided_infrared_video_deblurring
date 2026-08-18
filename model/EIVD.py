import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import conv1x1, conv3x3, conv5x5, actFunc


class dynamic_dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dynamic_dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        out = self.act(self.conv(x))
        out = out * self.scale
        out = torch.cat((x, out), 1)
        return out


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
        self.blur_perceiver = nn.Sequential(
            conv1x1(in_channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        blur_scale = self.blur_perceiver(x)[:, :, None, None]
        out = x

        for layer in self.layers:
            out = layer(out)

        out = self.conv1x1(out)
        out = x + blur_scale * out
        return out


class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
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


class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = DynamicGrowthRDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        x = self.rdb(x)
        out = self.down_sampling(x)
        return out


class EventFeatureExtractor(nn.Module):
    def __init__(self, para):
        super(EventFeatureExtractor, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B0 = conv5x5(2, self.n_feats, stride=1)
        self.time_encoder = conv1x1(1, self.n_feats)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)

    def forward(self, x, event_timestamps):
        out = self.F_B0(x)
        time_feat = self.time_encoder(event_timestamps)
        out = out + time_feat
        out = self.F_B1(out)
        out = self.F_B2(out)
        return out


class ImageFeatureExtractor(nn.Module):
    def __init__(self, para):
        super(ImageFeatureExtractor, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)

        self.blur_detector = nn.Sequential(
            conv3x3(3, self.n_feats // 2),
            actFunc(self.activation),
            conv3x3(self.n_feats // 2, 1),
            nn.Sigmoid()
        )

        self.blur_conv = conv1x1(1, self.n_feats)

    def forward(self, x):
        blur_map = self.blur_detector(x)
        out = self.F_B0(x)
        out = out + self.blur_conv(blur_map)
        out = self.F_B1(out)
        out = self.F_B2(out)
        return out, blur_map


class DDGAFusion(nn.Module):
    def __init__(self, para):
        super(DDGAFusion, self).__init__()
        self.n_feats = para.n_features
        self.img2evt_attn = nn.Sequential(
            conv1x1(4 * self.n_feats + 1, 2 * self.n_feats),
            actFunc(para.activation),
            conv1x1(2 * self.n_feats, 4 * self.n_feats),
            nn.Sigmoid()
        )

        self.evt2img_attn = nn.Sequential(
            conv1x1(4 * self.n_feats, 2 * self.n_feats),
            actFunc(para.activation),
            conv1x1(2 * self.n_feats, 4 * self.n_feats),
            nn.Sigmoid()
        )

        self.fusion_conv = conv3x3(4 * self.n_feats, 4 * self.n_feats)

    def forward(self, img_feat, event_feat, blur_map):
        blur_map = F.interpolate(blur_map, size=img_feat.shape[2:], mode='bilinear', align_corners=True)
        img_feat_with_blur = torch.cat([img_feat, blur_map], dim=1)
        evt_attn = self.img2evt_attn(img_feat_with_blur)
        weighted_evt = event_feat * evt_attn

        img_attn = self.evt2img_attn(event_feat)
        weighted_img = img_feat * img_attn

        fused = weighted_img + weighted_evt
        return self.fusion_conv(fused)


class GatedHiddenUpdate(nn.Module):
    def __init__(self, para):
        super(GatedHiddenUpdate, self).__init__()
        self.n_feats = para.n_features

        self.update_gate = nn.Sequential(
            conv3x3(5 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            conv3x3(5 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )
        self.new_state = nn.Sequential(
            conv3x3(5 * self.n_feats + self.n_feats, self.n_feats),
            DynamicGrowthRDB(in_channels=self.n_feats, base_growth=self.n_feats, num_layer=3),
            nn.Tanh()
        )

    def forward(self, x, s_prev):
        z = self.update_gate(x)
        r = self.reset_gate(x)
        s_new = self.new_state(torch.cat([x, r * s_prev], dim=1))
        s = (1 - z) * s_prev + z * s_new
        return s


class BGR(nn.Module):
    def __init__(self, para):
        super(BGR, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.F_R = RDNet(in_channels=5 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation).to(self.device)

        self.hidden_update = GatedHiddenUpdate(para).to(self.device)

        self.to(self.device)

    def forward(self, fused_feat, s_last):
        fused_feat = fused_feat.to(self.device)
        s_last = s_last.to(self.device)
        out = torch.cat([fused_feat, s_last], dim=1)
        out = self.F_R(out)
        s = self.hidden_update(out, s_last)

        return out, s


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
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        self.fusion = conv1x1(self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))

    def forward(self, hs):
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()
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
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_extractor = ImageFeatureExtractor(para).to(self.device)
        self.event_extractor = EventFeatureExtractor(para).to(self.device)
        self.attention_fusion = DDGAFusion(para).to(self.device)
        self.cell = BGR(para).to(self.device)
        self.recons = Reconstructor(para).to(self.device)
        self.fusion = GSA(para).to(self.device)

        self.to(self.device)

    def forward(self, x, profile_flag=False):
        x = [item.to(self.device) for item in x]

        if len(x) != 4:
            raise ValueError(f"The model expects 4 inputs, but received {len(x)} in practice.")

        blur_imgs, sharp_imgs, event_maps, event_timestamps = x

        if len(event_timestamps.shape) != 5:
            if len(event_timestamps.shape) == 4:
                event_timestamps = event_timestamps.unsqueeze(2)
            else:
                raise ValueError(f"The dimension of event timestamps is incorrect. Expected 5‑D, but got {len(event_timestamps.shape)}‑D.")

        if profile_flag:
            return self.profile_forward(x)

        outputs, forward_hs, backward_hs = [], [], []

        if len(blur_imgs.shape) > 5:
            blur_imgs = torch.squeeze(blur_imgs, dim=2)
        elif len(blur_imgs.shape) < 5:
            raise ValueError(f"blur_imgs has insufficient dimensions. Expected 5‑D, but got {len(blur_imgs.shape)}‑D.")
        batch_size, frames, channels, height, width = blur_imgs.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)

        s_forward = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            img_feat, blur_map = self.image_extractor(blur_imgs[:, i, :, :, :])
            event_feat = self.event_extractor(event_maps[:, i, :, :, :], event_timestamps[:, i, :, :, :])
            fused_feat = self.attention_fusion(img_feat, event_feat, blur_map)
            h, s_forward = self.cell(fused_feat, s_forward)
            forward_hs.append(h)

        s_backward = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in reversed(range(frames)):
            img_feat, blur_map = self.image_extractor(blur_imgs[:, i, :, :, :])
            event_feat = self.event_extractor(event_maps[:, i, :, :, :], event_timestamps[:, i, :, :, :])
            fused_feat = self.attention_fusion(img_feat, event_feat, blur_map)
            h, s_backward = self.cell(fused_feat, s_backward)
            backward_hs.append(h)
        backward_hs = backward_hs[::-1]

        hs = [forward_hs[i] + backward_hs[i] for i in range(frames)]

        for i in range(self.num_fb, frames - self.num_ff):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)

    def profile_forward(self, x):
        blur_imgs, sharp_imgs, event_maps, event_timestamps = x
        outputs, hs = [], []
        batch_size, frames, channels, height, width = blur_imgs.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)

        for i in range(frames):
            img_feat, blur_map = self.image_extractor(blur_imgs[:, i, :, :, :])
            event_feat = self.event_extractor(event_maps[:, i, :, :, :], event_timestamps[:, i, :, :, :])
            fused_feat = self.attention_fusion(img_feat, event_feat, blur_map)
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
    blur_imgs = iter_samples[0]
    if len(blur_imgs.shape) != 5:
        raise ValueError(f"Dimension error for blur_imgs in feed function. Expected 5‑D, but got {len(blur_imgs.shape)}‑D.")
    outputs = model(iter_samples)
    return outputs


def cost_profile(model, H, W, seq_length):
    device = next(model.parameters()).device
    blur_imgs = torch.randn(1, seq_length, 3, H, W).to(device)
    sharp_imgs = torch.randn(1, seq_length, 3, H, W).to(device)
    event_maps = torch.randn(1, seq_length, 2, H, W).to(device)
    event_timestamps = torch.randn(1, seq_length, 1, H, W).to(device)

    x = [blur_imgs, sharp_imgs, event_maps, event_timestamps]
    profile_flag = True
    model = model.to(device)
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params
