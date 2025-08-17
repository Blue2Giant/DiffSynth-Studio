## 解析一下WanVideo的VAE
我的工作目录：/inspire/hdd/global_user/yeziqi-240108100047/ljh/DiffSynth-Studio

DiffSynth-Studio/diffsynth/models/wan_video_vae.py
源码位置

输入视频 x：形状 [B, C_in, T, H, W]，通常 C_in=3（RGB）。latent 通道 z_dim：这里是 16。
时域下采样因子：由 temperal_downsample=[False, True, True] 决定，总因子=2×2=4。也就是说，编码器把 T 帧压到 T_latent=ceil(T/4) 个时间步的潜变量。

forward函数中：
编码器输出 z_dim*2 个通道（用于 mu 和 log_var），之后 conv1 保持通道不变，再 chunk 成两半；
解码前的 conv2 只是 1×1×1 的通道整形（这里 in/out 都是 z_dim）
CausalConv3d(k=1)：核是 1×1×1，形状不变；“因果”是为了时间因果性 + 和缓存配合做流式/增量编码解码。

vae的输入输出都必须显地输入scale
```python
    mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

    def single_encode(self, video, device):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x


    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)
```
在3DVAE中CausualConv3d的卷积核都是3，padding都是1，卷积后的时间维度不会变化，对于kernel和padding输入的是int的话，对所有维度都是使用这个尺度，输入输出的时空都不会发生变化
a single int – in which case the same value is used for the depth, height and width dimension

在因果3D卷积核中，padding的方式是前向padding
```python
class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """
    #一般的padding内容如下：pad = (W_left, W_right, H_left, H_right, D_left, D_right)，D表示的是时间维度，他修改成了时间上前面padding两倍，后面padding0
    #若提供了真实历史 cache_x（例如上一次输入的末尾几帧），就把它拼到当前 batch 前面，用真实数据替代部分/全部的左侧 pad，避免信息损失并减少无意义的零填充。

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)# 在时间维把历史帧拼到当前前面
            x = torch.cat([cache_x, x], dim=2)# 用缓存抵消一部分左侧 pad
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        
        return super().forward(x)
```

## ResidualBlock
VAE的encoder和decoder中都有ResidualBlock
里面包含了因果3D卷积，以及RMS_norm,RMS其实就是在某个维度上归一化
```python
class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias
```
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                #取出将来要缓存的尾部帧，这里是2
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                #如果这里可缓存的帧数不足2而上一块的缓存存在
                #就把上一块缓存的最后一帧补最前面，
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                # 4) 更新缓存：把“当前块的尾部（可能补齐到至少2帧）”存到 feat_cache[idx]，因此他的长度始终为1，写成列表是方便原地赋值操作
                feat_cache[idx] = cache_x
                # 同一个 residual 里可能有多个 CausalConv3d，用 idx 区分它们的各自缓存槽位
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h
```
比较费解的是这个这个cache的list分块是怎么来的，还有这个id
在VAE的encode的部分时候就有了
```python
    out = self.encoder(x[:, :, :1, :, :],
    feat_cache=self._enc_feat_map,
    feat_idx=self._enc_conv_idx)
    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    ## default的encode和decode的cache数量不一样，一个26，一个33
    # len(self._feat_map)
    # 33
    # self._enc_conv_num 
    # 26
```

## encode函数
```python
#假设输入的是x = [b,c,d,h,w]
    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        for i in range(iter_):
            self._enc_conv_idx = [0]
            #在时间维度上每4帧作为一个块，进行编码
            #在self._enc_feat_map 是一个28长度的列表，用来存causualconv3d的中间结果
            #在encoder的forward函数中feat_idx[0] 每遇到一个 CausalConv3d 就加 1，idx 就能映射到 feat_cache 的正确位置。
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :],
                                   feat_cache=self._enc_feat_map,
                                   feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                    feat_cache=self._enc_feat_map,
                                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            #scale[0]是均值，scale[1]是方差，在通道维度上进行缩放
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu
```