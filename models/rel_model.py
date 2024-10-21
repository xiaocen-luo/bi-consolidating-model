import numpy as np
from torch import nn
import math
from transformers import *
import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Linear(in_planes, in_planes, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LayerNorm2d(nn.Module):
    def __init__(self,
                 embed_dim,
                 eps=1e-6,
                 data_format="channels_first") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.parameter.Parameter(torch.ones(embed_dim))
        self.bias = nn.parameter.Parameter(torch.zeros(embed_dim))

        self.eps = eps
        self.data_format = data_format

        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (embed_dim, )

    def forward(self, x):
        if self.data_format == "channels_last":  # N,H,W,C
            return F.layer_norm(x, self.embed_dim, self.weight, self.bias,
                                self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)  # N,C,H,W

            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class BiConsolidatingModel(nn.Module):
    def __init__(self, inplanes, outplanes, config, reduction=16):

        super(BiConsolidatingModel, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        # Local Consolidation
        pdcs = config_model(config.config)
        self.init_block = Conv2d(pdcs[0], self.inplanes, self.outplanes, kernel_size=3, padding=1)
        self.block_1 = PDCBlock(pdcs[1], self.outplanes, self.outplanes)
        self.block_2 = PDCBlock(pdcs[2], self.outplanes, self.outplanes)
        self.block_3 = PDCBlock(pdcs[3], self.outplanes, self.outplanes)
        self.activation = nn.GELU()
        self.ln1 = LayerNorm2d(self.outplanes)
        self.ln2 = LayerNorm2d(self.outplanes)
        self.ln3 = LayerNorm2d(self.outplanes)
        self.ln4 = LayerNorm2d(self.outplanes)
        # Global Consolidation
        self.ca = ChannelAttention(outplanes, ratio=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        # [B, L, L, D]
        x = x.permute(0, 3, 1, 2)
        residual =x

        out = self.init_block(x)
        out = self.ln1(out)
        out = self.activation(out)

        out = self.block_1(out)
        out = self.ln2(out)
        out = self.activation(out)

        out = self.block_2(out)
        out = self.ln3(out)
        out = self.activation(out)

        out = self.block_3(out)
        out = self.ln4(out)

        out += residual
        out = self.activation(out)

        ca_out = self.ca(out) * out
        sa_out = self.sa(out) * out

        out = torch.cat([sa_out, ca_out], dim=1)

        # [B, L, L, D]
        out = out.permute(0, 2, 3, 1)

        return out

class RTEModel(nn.Module):
    def __init__(self, config):
        super(RTEModel, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim
        self.bert_encoder = BertModel.from_pretrained("bert-base-cased", cache_dir='./pre_trained_bert')
        self.relation_matrix = nn.Linear(self.bert_dim * 3, self.config.rel_num * self.config.tag_size)
        self.projection_matrix_up = nn.Linear(self.bert_dim * 2, self.bert_dim * 3)
        self.dis_embs = nn.Embedding(20, 20)
        self.embedding_proj = nn.Linear(self.bert_dim * 2 + 20 + 144, self.bert_dim)
        # self.embedding_proj = nn.Linear(self.bert_dim + 20 + 144, self.bert_dim)

        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.dropout_2 = nn.Dropout(self.config.entity_pair_dropout)
        self.activation = nn.GELU()
        self.elu = nn.ELU()
        self.biConsolidatingModel = BiConsolidatingModel(self.bert_dim * 1, self.bert_dim * 1, config)

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        bert_embs = self.bert_encoder(input_ids=token_ids, attention_mask=mask, output_attentions=True)
        encoded_text = bert_embs[0]
        attention_state = bert_embs[2]
        return encoded_text, attention_state

    def triple_score_matrix(self, encoded_text, attention_state, dist_inputs, train = True):
        # encoded_text: [batch_size, seq_len, bert_dim(768)] 1,2,3

        batch_size, seq_len, bert_dim = encoded_text.size()

        # attention embedding
        attention = None
        for _item in attention_state:
            if attention == None:
                attention = _item
                attention = attention.permute(0, 2, 3, 1)
            else:
                attention = torch.cat([attention, _item.permute(0, 2, 3, 1)], dim=-1)

        attention_emb = attention[:,:seq_len,:seq_len,:]
        # self-cross embedding
        m_s = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim)
        m_o = encoded_text.unsqueeze(1).repeat(1, seq_len, 1, 1)
        # m_so = self.elu(encoded_text.unsqueeze(2).repeat(1, 1, seq_len, 1) * encoded_text.unsqueeze(1).repeat(1, seq_len, 1, 1))  # BLL 2H
        # position embedding
        dis_emb = self.dis_embs(dist_inputs)
        entity_pairs = torch.cat([m_s, m_o, dis_emb, attention_emb], dim=-1)
        # entity_pairs = torch.cat([m_so, dis_emb, attention_emb], dim=-1)
        entity_pairs = self.elu(self.embedding_proj(entity_pairs))
        entity_pairs = entity_pairs.reshape(batch_size, seq_len, seq_len, bert_dim)


        entity_pairs = self.biConsolidatingModel(entity_pairs)
        entity_pairs = entity_pairs.reshape(batch_size, seq_len * seq_len, bert_dim * 2)
        entity_pairs = self.projection_matrix_up(entity_pairs)
        entity_pairs = self.dropout_2(entity_pairs)
        entity_pairs = self.activation(entity_pairs)
        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len, self.config.rel_num, self.config.tag_size)

        if train:
            # [batch_size, tag_size, rel_num, seq_len, seq_len]
            triple_scores = triple_scores.permute(0, 3, 1, 2, 4)
            return triple_scores.permute(0, 4, 1, 2, 3)

        else:
            # [batch_size, seq_len, seq_len, rel_num]
            return triple_scores.argmax(dim = -1).permute(0,3,1,2)


    def forward(self, data, train = True):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        dist_inputs = data['dist_inputs']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text, attention_state = self.get_encoded_text(token_ids, mask)
        encoded_text = self.dropout(encoded_text)
        # [batch_size, rel_num, seq_len, seq_len]
        output = self.triple_score_matrix(encoded_text, attention_state, dist_inputs, train)

        return output


# PDC
nets = {
    'baseline': {
        'layer0':  'cd',
        'layer1':  'ad',
        'layer2':  'rd',
        'layer3':  'cv',
        },
    'c': {
        'layer0':  'cd',
        },
    'a': {
        'layer0': 'ad-r',
    },
    'r': {
        'layer0': 'rd',
    },
    'c2': {
        'layer0':  'cd-xy',
        'layer1':  'cd-d',
        },
    'cc': {
        'layer0': 'cd',
        'layer1': 'cd',
    },
    'a2': {
        'layer0':  'ad',
        'layer1':  'ad-r',
        },
    'aa': {
        'layer0': 'ad',
        'layer1': 'ad',
    },
    'r2': {
        'layer0':  'rd-xy',
        'layer1':  'rd-d',
        },
    'rr': {
        'layer0': 'rd',
        'layer1': 'rd',
    },
    'caa': {
        'layer0': 'cd',
        'layer1': 'ad',
        'layer2': 'ad-r',
    },
    'c2a2': {
        'layer0':  'cd-xy',
        'layer1':  'cd-d',
        'layer2':  'ad',
        'layer3':  'ad-r',
        },
    'a2r2': {
        'layer0':  'ad',
        'layer1':  'ad-r',
        'layer2':  'rd-xy',
        'layer3':  'rd-d',
        },
    'c4': {
        'layer0':  'cd',
        'layer1':  'cd',
        'layer2':  'cd',
        'layer3':  'cd',
        },
    'a4': {
        'layer0': 'ad',
        'layer1': 'ad',
        'layer2': 'ad',
        'layer3': 'ad',
    },
    'a4-r': {
        'layer0': 'ad-r',
        'layer1': 'ad-r',
        'layer2': 'ad-r',
        'layer3': 'ad-r',
    },
    'r4': {
        'layer0': 'rd',
        'layer1': 'rd',
        'layer2': 'rd',
        'layer3': 'rd',
    },
    'c2c2': {
        'layer0':  'cd-xy',
        'layer1':  'cd-d',
        'layer2':  'cd-xy',
        'layer3':  'cd-d',
        },
    'r2r2': {
        'layer0':  'rd-xy',
        'layer1':  'rd-d',
        'layer2':  'rd-xy',
        'layer3':  'rd-d',
        },
    'caar-xy': {
        'layer0':  'cd-xy',
        'layer1':  'ad',
        'layer2':  'ad-r',
        'layer3':  'rd-xy',
        },
    'caar-d': {
        'layer0':  'cd-d',
        'layer1':  'ad',
        'layer2':  'ad-r',
        'layer3':  'rd-d',
        },
    'ccav': {
        'layer0': 'cd-xy',
        'layer1': 'cd-d',
        'layer2': 'ad',
        'layer3': 'cv',
    },
    'ccav-r': {
        'layer0': 'cd-xy',
        'layer1': 'cd-d',
        'layer2': 'ad-r',
        'layer3': 'cv',
    },
    'ccrv-xy': {
        'layer0': 'cd-xy',
        'layer1': 'cd-d',
        'layer2': 'rd-xy',
        'layer3': 'cv',
    },
    'ccrv-d': {
        'layer0': 'cd-xy',
        'layer1': 'cd-d',
        'layer2': 'rd-d',
        'layer3': 'cv',
    },
    'car2': {
        'layer0': 'cd',
        'layer1': 'ad-r',
        'layer2': 'rd-xy',
        'layer3': 'rd-d',
    },
    'carr': {
        'layer0': 'cd',
        'layer1': 'ad-r',
        'layer2': 'rd',
        'layer3': 'rd',
    },
    'cca': {
        'layer0': 'cd',
        'layer1': 'cd',
        'layer2': 'ad-r',
    },
    'car': {
        'layer0': 'cd',
        'layer1': 'ad-r',
        'layer2': 'rd',
    },
    'r4': {
        'layer0': 'rd',
        'layer1': 'rd',
        'layer2': 'rd',
        'layer3': 'rd',
    },
    'a2a2': {
        'layer0': 'ad',
        'layer1': 'ad-r',
        'layer2': 'ad',
        'layer3': 'ad-r',
    },
    'c2r2': {
        'layer0':  'cd-xy',
        'layer1':  'cd-d',
        'layer2':  'rd-xy',
        'layer3':  'rd-d',
        },
    'c2ar-r': {
        'layer0': 'cd-xy',
        'layer1': 'cd-d',
        'layer2': 'ad-r',
        'layer3': 'rd',
    },
    'ccar': {
        'layer0': 'cd',
        'layer1': 'cd',
        'layer2': 'ad-r',
        'layer3': 'rd',
    },
    'c2ar': {
        'layer0': 'cd-xy',
        'layer1': 'cd-d',
        'layer2': 'ad-r',
        'layer3': 'rd',
    },
    'ca2r': {
        'layer0': 'cd',
        'layer1': 'ad',
        'layer2': 'ad-r',
        'layer3': 'rd',
    },
    'caar': {
        'layer0': 'cd',
        'layer1': 'ad-r',
        'layer2': 'ad-r',
        'layer3': 'rd',
    },

    'c_xyarv': {
        'layer0': 'cd-xy',
        'layer1': 'ad',
        'layer2': 'rd',
        'layer3': 'cv',
    },
    'c_darv': {
        'layer0': 'cd-d',
        'layer1': 'ad',
        'layer2': 'rd',
        'layer3': 'cv',
    },
    'carv': {
        'layer0': 'cd',
        'layer1': 'ad-r',
        'layer2': 'rd',
        'layer3': 'cv',
    },
    'car_xyv': {
        'layer0': 'cd',
        'layer1': 'ad',
        'layer2': 'rd-xy',
        'layer3': 'cv',
    },
    'car_dv': {
        'layer0': 'cd',
        'layer1': 'ad',
        'layer2': 'rd-d',
        'layer3': 'cv',
    },
    'varv': {
        'layer0': 'cv',
        'layer1': 'ad',
        'layer2': 'rd',
        'layer3': 'cv',
    },
    'cvrv': {
        'layer0': 'cd',
        'layer1': 'cv',
        'layer2': 'rd',
        'layer3': 'cv',
    },
    'cavv': {
        'layer0': 'cd',
        'layer1': 'ad',
        'layer2': 'cv',
        'layer3': 'cv',
    },
    'vvvv': {
        'layer0': 'cv',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
    },
    }

def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(4):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(createConvFunc(op))

    return pdcs

class Conv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

## cd, ad, rd convolutions
def createConvFunc(op_type):
    # assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'cd-xy':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 3).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 3 * 3)
            weights = weights.view(shape[0], shape[1], -1)

            buffer[:, :, [1, 3, 5, 7]] = weights[:, :, 5:]
            buffer = buffer.view(shape[0], shape[1], 3, 3)

            buffer_c = buffer.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, buffer_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            return y - yc
        return func
    elif op_type == 'cd-d':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 3).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 3 * 3)
            weights = weights.view(shape[0], shape[1], -1)

            buffer[:, :, [0, 2, 6, 8]] = weights[:, :, 5:]
            buffer = buffer.view(shape[0], shape[1], 3, 3)

            buffer_c = buffer.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, buffer_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            return y - yc

        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'ad-r':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [1, 2, 5, 0, 4, 8, 3, 6, 7]]).view(shape) # river clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd-xy':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [2, 10, 14, 22]] = weights[:, :, 5:]
            buffer[:, :, [7, 11, 13, 17]] = -weights[:, :, 5:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd-d':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 4, 20, 24]] = weights[:, :, 5:]
            buffer[:, :, [6, 8, 16, 18]] = -weights[:, :, 5:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None

class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride = stride

        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y
