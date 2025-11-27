The Annotated Transformer - 구현 노트
Harvard NLP의 The Annotated Transformer를 따라 공부하면서 정리
논문 "Attention is All You Need"를 코드로 한 줄씩 따라가며 구현
시작 전에 필요한 라이브러리:
pythonimport math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

1. Model Architecture
전체 Transformer 구조를 먼저 정의합니다.
pythonclass EncoderDecoder(nn.Module):
    """
    표준 Encoder-Decoder 아키텍처.
    Transformer의 최상위 모듈
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed    # source embedding
        self.tgt_embed = tgt_embed    # target embedding
        self.generator = generator    # output layer
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: source sequence
        tgt: target sequence
        src_mask: source mask (padding)
        tgt_mask: target mask (padding + future masking)
        """
        return self.decode(self.encode(src, src_mask), src_mask,
                          tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
   
Generator
디코더의 출력을 vocabulary 크기의 logits으로 변환합니다.
pythonclass Generator(nn.Module):
    """Linear + Softmax로 다음 단어 예측"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # log_softmax를 사용하면 수치적으로 더 안정적
        return log_softmax(self.proj(x), dim=-1)
3. Encoder와 Decoder Stacks
Encoder
pythondef clones(module, N):
    """같은 layer를 N개 복사 (독립적인 파라미터)"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """N개의 layer를 쌓은 encoder stack"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        """각 layer를 순차적으로 통과"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
왜 clones를 쓸까?
단순히 [layer] * N 하면 같은 객체를 N번 참조하게 된다. deepcopy를 써야 각 layer가 독립적인 파라미터를 가짐

Layer Normalization
pythonclass LayerNorm(nn.Module):
    """
    Layer Normalization.
    논문에서는 Post-LN이지만, 실제로는 Pre-LN이 더 안정적입니다.
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
Sublayer Connection
Residual connection + Layer Normalization + Dropout
pythonclass SublayerConnection(nn.Module):
    """
    Residual connection 다음에 layer norm 적용.
    코드 단순화를 위해 norm을 먼저 적용하는 Pre-LN 방식 사용.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """x + Dropout(Sublayer(LayerNorm(x)))"""
        return x + self.dropout(sublayer(self.norm(x)))
Encoder Layer
pythonclass EncoderLayer(nn.Module):
    """
    Encoder는 2개의 sub-layer로 구성:
    1) Multi-head self-attention
    2) Position-wise feed forward
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Self-attention 후 Feed Forward"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
Decoder
pythonclass Decoder(nn.Module):
    """N개의 layer를 쌓은 decoder stack"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
Decoder Layer
pythonclass DecoderLayer(nn.Module):
    """
    Decoder는 3개의 sub-layer로 구성:
    1) Masked multi-head self-attention
    2) Multi-head cross-attention (over encoder output)
    3) Position-wise feed forward
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn      # Masked self-attention
        self.src_attn = src_attn        # Cross-attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # 1) Self-attention (with future masking)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) Cross-attention (Q=decoder, K=V=encoder)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 3) Feed forward
        return self.sublayer[2](x, self.feed_forward)
        
lambda를 쓰는 이유?
sublayer가 함수를 받도록 설계되어 있어서, lambda로 감싸줌

Masking
디코더는 미래 단어를 볼 수 없어야 합니다.
pythondef subsequent_mask(size):
    """
    미래 위치를 mask하는 상삼각 행렬.
    
    예: size=3
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
3. Attention
Scaled Dot-Product Attention
Transformer의 핵심 메커니즘입니다.
pythondef attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention 계산
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    d_k = query.size(-1)
    
    # 1) QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2) Masking (선택)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3) Softmax
    p_attn = scores.softmax(dim=-1)
    
    # 4) Dropout (선택)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # 5) Attention weights와 value 곱하기
    return torch.matmul(p_attn, value), p_attn
왜 sqrt(d_k)로 나누나?
d_k가 커지면 내적값이 너무 커져서 softmax가 극단값으로 수렴합니다.
Gradient가 거의 0이 되는 문제를 방지하기 위해 scaling합니다.
Multi-Head Attention
pythonclass MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        h: head 개수
        d_model: 모델 차원 (d_model = h * d_k)
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        
        # d_k = d_v = d_model / h 로 가정
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 모든 head에 같은 mask 적용
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Linear projection을 batch로 처리
        #    d_model => h x d_k로 reshape
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Attention 계산 (batch 처리)
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        
        # 3) "Concat" - transpose와 contiguous로 처리
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        
        # 4) 최종 linear projection
        del query
        del key
        del value
        return self.linears[-1](x)
핵심 트릭:

view와 transpose로 multi-head를 batch처럼 처리
4개의 linear: Q/K/V projection 3개 + 최종 output projection 1개

4. Position-wise Feed-Forward Networks
pythonclass PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    각 position에 독립적으로 적용되는 2-layer MLP.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
논문: d_model=512, d_ff=2048
5. Embeddings and Positional Encoding
Embeddings
pythonclass Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 논문에서 embedding weights에 sqrt(d_model) 곱함
        return self.lut(x) * math.sqrt(self.d_model)
왜 sqrt(d_model)을 곱하나?
Positional encoding과 scale을 맞추기 위해서입니다.
Positional Encoding
pythonclass PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional encoding 미리 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # buffer로 등록 (학습 안 됨, state_dict에는 저장됨)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
왜 sin/cos인가?

상대 위치를 선형 변환으로 표현 가능
학습 중 보지 못한 긴 sequence에도 extrapolate 가능

6. Full Model
이제 모든 조각을 조립합니다.
pythondef make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """표준 Transformer 모델 생성"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )
    
    # Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model
매개변수 설정 (Base model):

N = 6 (layers)
d_model = 512
d_ff = 2048
h = 8 (heads)
d_k = d_v = 64 (d_model / h)

7. Inference
Greedy decoding 예제:
pythondef greedy_decode(model, src, src_mask, max_len, start_symbol):
    """한 번에 하나씩 가장 확률 높은 단어 선택"""
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src.data).fill_(start_symbol)
    
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    
    return ys
배운 점

Residual Connection의 중요성:
깊은 네트워크 학습을 가능하게 한다.
Layer Normalization vs Batch Normalization:
Sequence 길이가 다양한 NLP에서는 LayerNorm이 더 적합
Multi-Head의 의미:
여러 representation subspace에서 정보를 추출
Masking의 종류:

Padding mask: 패딩 토큰 무시
Future mask: 미래 단어 참조 방지


Positional Encoding:
순서 정보가 없는 architecture에 위치 정보를 주입
