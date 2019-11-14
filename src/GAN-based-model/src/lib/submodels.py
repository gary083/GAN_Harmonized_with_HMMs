import torch
import torch.nn as nn
from IPython import embed
from .modules import Conv1dNorm, Highway, Prenet, ResBlock, MLP


class EncCBHL(nn.Module):
    def __init__(self, input_size, out_channels, proj_dim=128, K=15, projections=[128, 128]):
        super(EncCBHL, self).__init__()
        self.prenet = Prenet(input_size, hidden_dims=[256, proj_dim])
        self.cbhl = CBHL(proj_dim, out_channels, K, projections)
        self.feature_size =  int(out_channels.split('_')[-1])

    def forward(self, inputs, input_lengths):
        y = self.prenet(inputs)
        y = self.cbhl(y, input_lengths)
        return y


class CBHL(nn.Module): 
    '''
    CBHG-like 
    in_dim : input dimension 
    K: max kernel size --> there would be 1, 3, ....., to k//2 kernels
    '''
    def __init__(self, in_dim, out_channels, K=15, projections=[128, 128]):
        super(CBHL, self).__init__()
        self.in_dim = in_dim
        self.conv1d_banks = nn.ModuleList([
            Conv1dNorm(in_dim, in_dim, kernel_size=k, stride=1)
            for k in range(1, K+1, 2)])
        #self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        in_sizes = [(K+1)//2 * in_dim] + projections[:-1]
        
        self.conv1d_projections = nn.ModuleList([
            Conv1dNorm(in_size, out_size, kernel_size=3, stride=1)
            for in_size, out_size in zip(in_sizes, projections)])
        
        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList([
            Highway(in_dim, in_dim) for _ in range(4)])

        feature_size = int(out_channels.split('_')[-1])
        hidden = '_'.join(out_channels.split('_')[:-1])
        self.linear = MLP(in_dim, feature_size, hidden)

    def forward(self, inputs, input_lengths):
        #inputs  (bs, T, f)
        x = inputs 
        assert inputs.shape[-1] == self.in_dim
        x = x.permute(0, 2, 1)
        x = torch.cat([conv1d(x, input_lengths) for conv1d in self.conv1d_banks], dim=1)
        #assert x.shape[1] == self.in_dim * ((K+1) //2)
        #x = self.max_pool1d(x)
        for conv1d in self.conv1d_projections :
            x = conv1d(x, input_lengths)

        #back to original shape
        x = x.permute(0, 2, 1)

        x = self.pre_highway(x)

        #residual 
        x += inputs
        for highway in self.highways :
            x = highway(x)

        outputs = self.linear(x)
        return outputs


class VGGExtractor(nn.Module): 
    #modified from LAS extractor, except that T is not reduced 
    def __init__(self, in_dim):
        super(VGGExtractor, self).__init__()
        in_channel, freq_dim, out_dim = self.check_dim(in_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(2, stride=2) # Half-time dimension
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(2, stride=2) # Half-time dimension

    @staticmethod
    def check_dim(in_dim):
        d = in_dim
        if d % 13 == 0:
            # MFCC feature
            return d//13, 13, (13)*32
        elif d % 40 == 0:
            # Fbank feature
            return d//40, 40, (40)*32
        else:
            raise ValueError('Acoustic feature dimension for VGG should be k*13 (MFCC) or k*40 (Fbank) but got ' + d)

    def view_input(self, feature):
        # drop time
        # xlen /= 4
        # if feature.shape[1] % 4 != 0:
        #     feature = feature[:,:-(feature.shape[1] % 4),:].contiguous()
        bs, ts, ds = feature.shape
        # reshape
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1,2)
        return feature

    def forward(self, feature):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature = self.view_input(feature)
        feature = self.relu(self.conv1(feature))
        feature = self.relu(self.conv2(feature)) #BSx16xTxD
        feature = self.relu(self.conv3(feature))
        feature = self.relu(self.conv4(feature)) #BSx32xTxD
        # BSx32xTxD -> BSxTx32xD
        feature = feature.transpose(1,2)
        #  BS x T x 32 x D -> BS x T x 32D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], self.out_dim)
        return feature


class Compressor(nn.Module):
    def __init__(self, receptive_field, input_size=39, channels=None, kernel_size=None, activation='ReLU', **kwargs):
        super(Compressor, self).__init__()
        #TODO: add this to config
        assert input_size == 39
        in_channel, freq_dim, out_dim = self.check_dim(input_size)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        chs = list(map(int, channels.split('_')))
        ks = list(map(int, kernel_size.split('_')))
        assert len(chs) == len(ks)

        self.feature_size = (receptive_field//2//2) * chs[-1]

        blocks = []
        blocks.append(nn.Conv2d(in_channel,chs[0], ks[0], stride=1, padding=1))
        blocks.append(getattr(nn, activation)())
        blocks.append(nn.Conv2d(chs[0], chs[1], ks[1], stride=1, padding=1))
        blocks.append(getattr(nn, activation)())
        blocks.append(nn.MaxPool2d(2, stride=2))
        blocks.append(nn.Conv2d(chs[1], chs[2], ks[2], stride=1, padding=1))
        blocks.append(getattr(nn, activation)())
        blocks.append(nn.Conv2d(chs[2], chs[3], ks[3], stride=1, padding=1))
        blocks.append(getattr(nn, activation)())
        blocks.append(nn.MaxPool2d(2, stride=2))
        blocks.append(nn.Conv2d(chs[3], chs[4], ks[4], stride=1, padding=1))
        blocks.append(getattr(nn, activation)())
        blocks.append(nn.Conv2d(chs[4], chs[5], ks[5], stride=1, padding=1))
        blocks.append(getattr(nn, activation)())
        blocks.append(nn.MaxPool2d((1,2), stride=(1, 2)))
        self.model = nn.ModuleList(blocks)

    @staticmethod
    def check_dim(in_dim):
        d = in_dim
        if d % 13 == 0:
            # MFCC feature
            return d//13, 13, (13)*32
        else:
            raise ValueError('Acoustic feature dimension for VGG should be k*13 (MFCC) or k*40 (Fbank) but got ' + d)

    def view_input(self, feature):
        # drop time
        # xlen /= 4
        # if feature.shape[1] % 4 != 0:
        #     feature = feature[:,:-(feature.shape[1] % 4),:].contiguous()
        bs, ts, ds = feature.shape
        # reshape
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1,2)
        return feature

    def forward(self, feature, xlen=None):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature = self.view_input(feature)
        for block in self.model :
            feature = block(feature) 
        feature = feature.transpose(1,2)
        #  BS x T//2//2 x 32 x 1 -> BS x T//2//2 x 32
        feature = feature.contiguous().view(feature.shape[0], -1)
        return feature


class EncResBlock(nn.Module):
    #in_dim : feature dim
    #out_dim: ctc output token dim
    def __init__(self, input_size, channels, kernel_size, activation='ReLU', **kwargs):
        super(EncResBlock, self).__init__()
        self.extractor = VGGExtractor(input_size)
        #self.intermediate_i = kernel_size.split('x')[0].count('_')
        kernel_size = kernel_size.replace('x', '_')
        self.dims = list(map(int, channels.split('_')))
        self.ks = list(map(int, kernel_size.split('_')))
        blocks = []
        in_dim = self.extractor.out_dim #32x13
        for out_dim, k in zip(self.dims, self.ks):
            blocks.append(ResBlock(in_dim, out_dim, k, activation=getattr(nn, activation)()))
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)
        self.feature_size =  self.dims[-1]
        self.intermediate_feature_size = self.feature_size

    def forward(self, x, xlen=None):
        """
        Args:
            x: tensor with shape (batch_size, timesteps, input_size)
            input_length: the length of elements in x before padding
        Return:
            tensor with shape (batch_size, timesteps, output_size)
        """
        B, T, in_dim = x.shape
        y = self.extractor(x)
        y = y.permute(0, 2, 1)
        for block in self.blocks :
            y = block(y, xlen)
        y = y.permute(0 , 2, 1)
        return None, y


class Simple(nn.Module):
    #in_dim : feature dim
    #out_dim: ctc output token dim
    def __init__(self, input_size, kernel_size, channels, activation='ReLU', dropout_rate=0, **kwargs):
        super(Simple, self).__init__()
        self.intermediate_i = kernel_size.split('x')[0].count('_')
        kernel_size = kernel_size.replace('x' ,'_')
        KS = list(map(int, kernel_size.split('_')))
        CHS = list(map(int, channels.split('_')))
        assert len(KS) == len(CHS)
        in_channels = [input_size] + CHS[:-1]
        out_channels = CHS

        #self.activation_fn = getattr(nn, activation)()
        blocks = []
        for in_c, out_c, ks in zip(in_channels, out_channels, KS):
            blocks.append(Conv1dNorm(in_c, out_c, ks, stride=1, padding=None, activation=activation))
            #blocks.append(activation_fn)

        self.blocks = nn.ModuleList(blocks)
        self.feature_size = out_channels[-1]
        self.intermediate_feature_size = CHS[self.intermediate_i]

    def forward(self, x, xlen=None, log_scale=True):
        """
        Args:
            x: tensor with shape (batch_size, timesteps, input_size)
            input_length: the length of elements in x before padding
        Return:
            tensor with shape (batch_size, timesteps, output_size)
        """
        B, T, in_dim = x.shape
        y = x.permute(0, 2, 1)
        for i, block in enumerate(self.blocks) :
            y = block(y, xlen)
            #y = self.activation_fn(y)
            if i == self.intermediate_i :
                y1 = y.permute(0, 2, 1) 
        y = y.permute(0 , 2, 1)
        return y1, y
