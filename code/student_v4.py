"""Final self-contained GPT with a window cache and compact trigram backoff.

This file contains the complete final implementation so the submitted
checkpoint does not depend on intermediate student versions.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class ModernBlock(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.heads = heads
        self.dropout = dropout
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0., is_causal=True)
        x = x + F.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)),
                          p=self.dropout, training=self.training)
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        mlp = self.down(F.silu(gate) * up)
        return x + F.dropout(mlp, p=self.dropout, training=self.training)


def cache_distribution(features, ids, theta, vocab, information_weights=None, threshold=0.):
    attention, values = cache_attention(features, ids, theta, information_weights, threshold)
    batch, length = ids.shape
    result = features.new_zeros((batch, length, vocab), dtype=torch.float32)
    return result.scatter_add(-1, values[:, None, :].expand(batch, length, length), attention)


def cache_attention(features, ids, theta, information_weights=None, threshold=0.):
    batch, length = ids.shape
    normalized = F.normalize(features.float(), dim=-1)
    similarity = theta * (normalized @ normalized.transpose(-1, -2))
    positions = torch.arange(length, device=ids.device)
    allowed = positions[None, :] < positions[:, None]
    allowed[0, 0] = True
    attention = similarity.masked_fill(~allowed, -torch.inf).softmax(-1)
    attention = attention * (positions > 0)[None, :, None]
    values = torch.cat((ids[:, 1:], ids.new_zeros((batch, 1))), dim=1)
    if information_weights is not None and threshold > 0:
        selected = information_weights[values] >= threshold
        attention = attention * selected[:, None, :]
        mass = attention.sum(-1, keepdim=True)
        attention = torch.where(mass > 0, attention / mass.clamp_min(1e-30), attention)
    return attention, values


class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.width = config['width']
        self.heads = config['heads']
        self.dropout = float(config.get('dropout', 0.1))
        self.cache_alpha = float(config.get('cache_alpha', 0.))
        self.cache_theta = float(config.get('cache_theta', 10.))
        self.cache_mode = config.get('cache_mode', 'linear')
        self.cache_gamma = float(config.get('cache_gamma', 0.))
        self.cache_threshold = float(config.get('cache_threshold', 0.))
        weights = config.get('information_weights', [1.] * config['vocab'])
        if len(weights) != config['vocab']:
            raise ValueError('information_weights must match vocabulary size')
        self.register_buffer('information_weights', torch.tensor(weights, dtype=torch.float32), persistent=False)
        hidden = int(config.get('hidden', math.ceil((8 * self.width / 3) / 64) * 64))
        head_dim = self.width // self.heads
        if head_dim % 2:
            raise ValueError('head dimension must be even for RoPE')
        self.token = nn.Embedding(config['vocab'], self.width)
        self.blocks = nn.ModuleList([
            ModernBlock(self.width, self.heads, hidden, self.dropout)
            for _ in range(config['depth'])])
        self.norm = RMSNorm(self.width)
        self.head = nn.Linear(self.width, config['vocab'], bias=False)
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=.02 / math.sqrt(2 * config['depth']))
            nn.init.normal_(block.down.weight, std=.02 / math.sqrt(2 * config['depth']))
        self.head.weight = self.token.weight
        inv_freq = 1. / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(self.context).float()
        angles = torch.outer(positions, inv_freq)
        self.register_buffer('rope_cos', torch.cat((angles, angles), dim=-1).cos()[None, None], persistent=False)
        self.register_buffer('rope_sin', torch.cat((angles, angles), dim=-1).sin()[None, None], persistent=False)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def features(self, ids):
        length = ids.shape[1]
        x = F.dropout(self.token(ids), p=self.dropout, training=self.training)
        cos = self.rope_cos[:, :, :length].to(dtype=x.dtype)
        sin = self.rope_sin[:, :, :length].to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        features = self.features(ids)
        base = F.log_softmax(self.head(features).float(), dim=-1)
        cache_strength = self.cache_gamma if self.cache_mode == 'information' else self.cache_alpha
        if cache_strength <= 0 or ids.shape[1] < 2:
            return base
        if self.cache_mode == 'linear':
            attention, values = cache_attention(
                features, ids, self.cache_theta,
                self.information_weights, self.cache_threshold)
            probability = base.exp().mul_(1 - self.cache_alpha)
            probability.scatter_add_(
                -1, values[:, None, :].expand(ids.shape[0], ids.shape[1], ids.shape[1]),
                attention * self.cache_alpha)
            probability[:, 0] = base[:, 0].exp()
            if self.cache_threshold > 0:
                probability.div_(probability.sum(-1, keepdim=True))
            return probability.clamp_min_(1e-30).log_()
        cache = cache_distribution(
            features, ids, self.cache_theta, self.config['vocab'],
            self.information_weights, self.cache_threshold)
        if self.cache_mode == 'information':
            token_weight = (self.cache_gamma * self.information_weights).clamp(0, .999)
            probability = base.exp() * (1 - token_weight) + cache * token_weight
            mixed = probability.clamp_min(1e-30).log()
            mixed = mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)
        else:
            raise ValueError(f'Unknown cache mode: {self.cache_mode}')
        return torch.cat((base[:, :1], mixed[:, 1:]), dim=1)


class TrigramGPT(ModernGPT):
    def __init__(self, config):
        super().__init__(config)
        vocab = config['vocab']
        self.trigram_alpha = float(config.get('trigram_alpha', 0.))
        self.static_bigram_alpha = float(config.get('static_bigram_alpha', 0.))
        self.use_trigram = bool(config.get('use_trigram', False))
        if self.use_trigram:
            contexts = int(config['trigram_contexts'])
            top_k = int(config['trigram_top_k'])
            self.register_buffer('bigram_probability', torch.zeros(vocab, vocab))
            self.register_buffer('trigram_context_keys', torch.zeros(contexts, dtype=torch.int32))
            self.register_buffer('trigram_top_ids', torch.zeros(contexts, top_k, dtype=torch.int16))
            self.register_buffer('trigram_top_direct', torch.zeros(contexts, top_k, dtype=torch.float16))
            self.register_buffer('trigram_backoff', torch.zeros(contexts, dtype=torch.float16))

    def predict_log_probs(self, ids):
        if self.cache_mode != 'linear':
            raise ValueError('student_v4 supports the selected linear window cache only')
        features = self.features(ids)
        probability = self.head(features).float().softmax(-1)
        if self.cache_alpha > 0 and ids.shape[1] >= 2:
            attention, values = cache_attention(
                features, ids, self.cache_theta,
                self.information_weights, self.cache_threshold)
            first = probability[:, 0].clone()
            probability.mul_(1-self.cache_alpha)
            probability.scatter_add_(
                -1, values[:, None, :].expand(ids.shape[0], ids.shape[1], ids.shape[1]),
                attention*self.cache_alpha)
            probability[:, 0] = first
        if self.use_trigram and self.trigram_alpha > 0:
            ngram = self.bigram_probability[ids]
            coefficient = ngram.new_full(ids.shape, self.static_bigram_alpha + self.trigram_alpha)
            direct = top_ids = None
            if ids.shape[1] >= 2:
                contexts = (ids[:, :-1]*self.config['vocab'] + ids[:, 1:]).flatten()
                positions = torch.searchsorted(self.trigram_context_keys, contexts)
                positions.clamp_max_(len(self.trigram_context_keys)-1)
                found = (self.trigram_context_keys[positions] == contexts).reshape(ids.shape[0], -1)
                rows = positions.reshape(ids.shape[0], -1)
                top_ids = self.trigram_top_ids[rows].long()
                direct = self.trigram_top_direct[rows].float() * found.unsqueeze(-1)
                backoff = self.trigram_backoff[rows].float()
                coefficient[:, 1:] = torch.where(
                    found, self.static_bigram_alpha + self.trigram_alpha*backoff,
                    coefficient[:, 1:])
            probability.mul_(1-self.static_bigram_alpha-self.trigram_alpha)
            ngram.mul_(coefficient.unsqueeze(-1))
            probability.add_(ngram)
            if direct is not None:
                probability[:, 1:].scatter_add_(-1, top_ids, direct*self.trigram_alpha)
        return probability.clamp_min_(1e-30).log_()


def build_model(config):
    return TrigramGPT(config)
