import torch
import torch.nn as nn


class TrickShuffling(nn.Module):
    def __init__(self, window_size, in_channels):
        super().__init__()
        self.embed_dim = window_size ** 2 * 3
        self.patch_encoding = nn.Conv2d(in_channels,
                                        self.embed_dim,
                                        kernel_size=window_size,
                                        stride=window_size,
                                        bias=False)

    def forward(self, x):
        return self.patch_encoding(x)


class RandAutoencoder(nn.Module):
    def __init__(self, in_dim=256, channel=3, ratio=100):
        super().__init__()
        self.ratio = ratio
        self.random_encoder = TrickShuffling(64, 3)
        self.dropout = nn.Dropout(0.1)
        self.BN = nn.BatchNorm2d(channel)
        self.shape = None

    def transpose_multi_samples(self, encoding):
        encoding = encoding.reshape(self.shape)
        return encoding

    def forward(self, neg_samples):
        self.shape = neg_samples.shape
        neg_samples = neg_samples.repeat(1, 1, 1, self.ratio)
        auto_encodings = self.random_encoder(neg_samples).chunk(self.ratio, -1)
        encodings = []
        with torch.no_grad():
            for i, encoding in enumerate(auto_encodings):
                encodings.append(self.transpose_multi_samples(encoding))
            auto_encodings = torch.cat(encodings, 3)
            auto_encodings = self.dropout(auto_encodings)
            auto_encodings = self.BN(auto_encodings)
            sets = torch.cat(auto_encodings.chunk(self.ratio, -1), 0)
        return sets


if __name__ == '__main__':
    t = torch.randn([2, 3, 256, 256])
    model = RandAutoencoder(in_dim=256, channel=3, ratio=3)
    out = model(t)
    print(out.shape)
