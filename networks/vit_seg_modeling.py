# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
from torch.nn.modules.utils import _pair
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2



logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
class AttentionExchange(nn.Module):
    def __init__(self, config, vis):
        super(AttentionExchange, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.gaze_query = Linear(config.hidden_size, self.all_head_size)
        self.gaze_key = Linear(config.hidden_size, self.all_head_size)
        self.gaze_value = Linear(config.hidden_size, self.all_head_size)

        self.gaze_out = Linear(config.hidden_size, config.hidden_size)
        self.gaze_attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.gaze_proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,gaze):
        ##img作为Q gaze作为K和V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(gaze)
        mixed_value_layer = self.value(gaze)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        #gaze作为Q，img作为K和V
        gaze_mixed_query_layer = self.gaze_query(gaze)
        gaze_mixed_key_layer = self.gaze_key(hidden_states)
        gaze_mixed_value_layer = self.gaze_value(hidden_states)

        gaze_query_layer = self.transpose_for_scores(gaze_mixed_query_layer)
        gaze_key_layer = self.transpose_for_scores(gaze_mixed_key_layer)
        gaze_value_layer = self.transpose_for_scores(gaze_mixed_value_layer)

        gaze_attention_scores = torch.matmul(gaze_query_layer, gaze_key_layer.transpose(-1, -2))
        gaze_attention_scores = gaze_attention_scores / math.sqrt(self.attention_head_size)
        gaze_attention_probs = self.softmax(gaze_attention_scores)
        weights = gaze_attention_probs if self.vis else None
        gaze_attention_probs = self.gaze_attn_dropout(gaze_attention_probs)

        gaze_context_layer = torch.matmul(gaze_attention_probs, gaze_value_layer)
        gaze_context_layer = gaze_context_layer.permute(0, 2, 1, 3).contiguous()
        gaze_new_context_layer_shape = gaze_context_layer.size()[:-2] + (self.all_head_size,)
        gaze_context_layer = gaze_context_layer.view(*gaze_new_context_layer_shape)
        gaze_attention_output = self.gaze_out(gaze_context_layer)
        gaze_attention_output = self.gaze_proj_dropout(gaze_attention_output)


        return attention_output, weights, gaze_attention_output

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.gaze_query = Linear(config.hidden_size, self.all_head_size)
        self.gaze_key = Linear(config.hidden_size, self.all_head_size)
        self.gaze_value = Linear(config.hidden_size, self.all_head_size)

        self.gaze_out = Linear(config.hidden_size, config.hidden_size)
        self.gaze_attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.gaze_proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,gaze):
        ##img作为Q gaze作为K和V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(gaze)
        mixed_value_layer = self.value(gaze)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        # #gaze作为Q，img作为K和V
        gaze_mixed_query_layer = self.gaze_query(gaze)
        gaze_mixed_key_layer = self.gaze_key(hidden_states)
        gaze_mixed_value_layer = self.gaze_value(hidden_states)

        gaze_query_layer = self.transpose_for_scores(gaze_mixed_query_layer)
        gaze_key_layer = self.transpose_for_scores(gaze_mixed_key_layer)
        gaze_value_layer = self.transpose_for_scores(gaze_mixed_value_layer)

        gaze_attention_scores = torch.matmul(gaze_query_layer, gaze_key_layer.transpose(-1, -2))
        gaze_attention_scores = gaze_attention_scores / math.sqrt(self.attention_head_size)
        gaze_attention_probs = self.softmax(gaze_attention_scores)
        weights = gaze_attention_probs if self.vis else None
        gaze_attention_probs = self.gaze_attn_dropout(gaze_attention_probs)

        gaze_context_layer = torch.matmul(gaze_attention_probs, gaze_value_layer)
        gaze_context_layer = gaze_context_layer.permute(0, 2, 1, 3).contiguous()
        gaze_new_context_layer_shape = gaze_context_layer.size()[:-2] + (self.all_head_size,)
        gaze_context_layer = gaze_context_layer.view(*gaze_new_context_layer_shape)
        gaze_attention_output = self.gaze_out(gaze_context_layer)
        gaze_attention_output = self.gaze_proj_dropout(gaze_attention_output)


        return attention_output, weights,gaze_attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config

        self.con=Conv2dReLU(in_channels=2048,out_channels=1024,kernel_size=1)
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

        self.gaze_patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.gaze_position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.gaze_dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x,gaze,ac):
        if self.hybrid:
            x, features,gaze = self.hybrid_model(x,gaze,ac)
        else:
            features = None

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)


        gaze = self.gaze_patch_embeddings(gaze)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        gaze = gaze.flatten(2)
        gaze = gaze.transpose(-1, -2)  # (B, n_patches, hidden)

        gaze_embeddings = gaze + self.gaze_position_embeddings
        gaze_embeddings = self.gaze_dropout(gaze_embeddings)
        return embeddings, features, gaze_embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.gaze_attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.gaze_ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.gaze_ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x,y):
        h = x
        g = y
        x = self.attention_norm(x)
        y = self.gaze_attention_norm(y)
        x, weights,y = self.attn(x,y)
        x = x + h
        y = y+g

        h = x
        g = y
        x = self.ffn_norm(x)
        y = self.gaze_ffn_norm(y)
        x = self.ffn(x)
        y = self.gaze_ffn(y)
        x = x + h
        y = y + g
        return x, weights,y





class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.gaze_encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states,gaze):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights,gaze= layer_block(hidden_states,gaze)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        gaze_encoded=self.gaze_encoder_norm(gaze)
        return encoded, attn_weights,gaze_encoded


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)


    def forward(self, input_ids,gaze,ac):
        embedding_output, features, embedding_gaze = self.embeddings(input_ids,gaze,ac)
        encoded, attn_weights,gaze_encoded= self.encoder(embedding_output,embedding_gaze)  # (B, n_patch, hidden)
        return encoded, attn_weights, features,gaze_encoded

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv_more1 = Conv2dReLU(
            config.hidden_size * 2,
            config.hidden_size,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.subfusion = SupDecoder()

    def forward(self, hidden_states,gaze_attn,ll,  features=None,):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        y = gaze_attn.permute(0, 2, 1)
        y = y.contiguous().view(B, hidden, h, w)
        con = torch.cat([x, y], dim=1)
        x = self.conv_more1(con)
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            if i == 3:
                x = self.subfusion(x, ll)
            x = decoder_block(x, skip=skip)
        return x

class SupDecoder(nn.Module):
    def __init__(self):
        super(SupDecoder, self).__init__()
        self.conv1=Conv2dReLU(in_channels=3,out_channels=64,kernel_size=3,padding=1)
        self.conv2=Conv2dReLU(in_channels=128,out_channels=64,kernel_size=1)
    def forward(self,x,y):
        
        if y.size()[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        img=x
        y=self.conv1(y)
        con=torch.cat([x,y],dim=1)
        x=self.conv2(con)
        x=img+x
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x,gaze,ac,dc):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        if gaze.size()[1] == 1:
            gaze = gaze.repeat(1,3,1,1)
        x,  attn_weights,features,gaze_encoded = self.transformer(x,gaze,dc)  # (B, n_patch, hidden)

        x = self.decoder(x,gaze_encoded,ac,features)
        logits = self.segmentation_head(x)
        return logits

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


