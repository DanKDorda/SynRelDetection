import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data.proposal_dataset import ProposalDs
import math


class FinalPredictor(nn.Module):
    def __init__(self, opts):
        super(FinalPredictor, self).__init__()
        self.opts = opts
        self.n_proposals = 4

        # NETWORK STRUCTURE
        self.preliminary_transform = nn.Linear(3, 256)

        self.feat_out = 256
        # self.final_layer = nn.Linear(256, self.feat_out)
        n_transforms = self.opts.FP.n_transforms
        hidden = 256
        n_heads = 4
        dropout = 0.1
        self.feat_transformer_blocks = nn.ModuleList([TransformerBlock(hidden, n_heads, hidden*4, dropout) for _ in range(n_transforms)])

    def forward(self, position_tensor, orientation_tensor, adjacency_tensor, chosen_idx, proposals):
        """
        :param position_tensor:
        :param orientation_tensor:
        :param adjacency_tensor:
        :param chosen_idx:
        :param proposals: batch x n_props x 1 = (4, 4, 1)
        :param target:
        :return: pre_softmax_scores: batch, n_props, 1
        """
        # select the relevant relationship row
        relevant_relationships = adjacency_tensor[:, chosen_idx]
        # relevant_relationships.unsqueeze_(3)
        # use a TRANSFORMER to create an attention pooling thing which creates one output vector for all the connected components
        # the mask is the weight vector from the adjacency
        n_objects = 10

        # size: batch x n_object x 3
        full_features = torch.cat([position_tensor, orientation_tensor], dim=2)
        # obfuscate the chosen angle
        full_features[:, chosen_idx, 2] = 0
        full_features = self.preliminary_transform(full_features)

        mask = torch.ones(self.opts.batch_size, n_objects)
        mask[:, chosen_idx] = 0

        # where to use relevant_relationships???

        scoring_vector = full_features #torch.empty(self.opts.batch_size, )

        for transformer in self.feat_transformer_blocks:
            scoring_vector = transformer.forward(scoring_vector, None)

        # size: batch x n_proposal x 1
        target_pos = position_tensor[:, chosen_idx].unsqueeze(1)
        target_pos = target_pos.repeat(1, 4, 1)
        embed_proposals = self.preliminary_transform(torch.cat([target_pos, proposals], dim=2))
        important_scores = scoring_vector[:, chosen_idx].unsqueeze(1).transpose(-2, -1)
        embeddings = embed_proposals
        pre_softmax_scores = torch.matmul(embeddings, important_scores)/math.sqrt(embeddings.size(-1))
        return pre_softmax_scores


########################################################################################
# Transformer code from: https://github.com/codertimo/BERT-pytorch
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


if __name__ == '__main__':
    print('testing fp')
    from easydict import EasyDict as edict
    from PIL import Image

    opts = {'batch_size': 1}
    opts = edict(opts)


    def peek_at_im(tensor):
        Image.fromarray((np.array(tensor * 255).astype('uint8')).transpose(1, 2, 0)).show()


    fp = FinalPredictor(opts)
    objects = [[{'id': 0, 'orientation': 0.3132}]]
    chosen_idx = [0]
    props = fp.generate_image_proposals(objects, chosen_idx)
    # peek_at_im(props[0, 0])
    # peek_at_im(props[0, 1])
    # peek_at_im(props[0, 2])
    # peek_at_im(props[0, 3])
    print(props.shape)
    print('finished')
