# Transformer 구조 이해하기

"Attention is All You Need" 논문을 읽고 코드로 구현.

## 기본 아이디어

**기존 방식 (RNN)**: 문장을 앞에서부터 순서대로 읽음 → 느리고, 앞 내용을 까먹음

**Transformer**: 문장 전체를 한번에 보고, "어느 단어에 집중할지" 계산 → 빠르고, 문맥 파악 잘됨

## 전체 구조

```
영어 문장 → [Encoder] → 문장의 의미 파악
                           ↓
프랑스어   ← [Decoder] ← 의미를 기반으로 번역
```

Encoder와 Decoder 모두 같은 블록을 여러 번(6번) 쌓은 구조입니다.

## 핵심 구성요소

### 1. Attention - "어디를 볼까?"

```python
def attention(query, key, value):
    # 1. query와 key를 비교해서 관련도 계산
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)  # 너무 크면 안정성 떨어짐
    
    # 2. 관련도를 확률로 변환
    attention_weights = F.softmax(scores, dim=-1)
    
    # 3. value에 가중치 곱해서 결과 만들기
    output = torch.matmul(attention_weights, value)
    return output
```

### 2. Multi-Head Attention - "여러 관점에서 보기"

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        self.h = h  # head 개수 (보통 8개)
        self.d_k = d_model // h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
    
    def forward(self, query, key, value):
        # 1. Q, K, V를 h개로 쪼갬
        # 2. 각각 attention 계산
        # 3. 결과를 다시 합침
        ...
```

**왜 필요한가?**
- 한 번의 attention으로는 한 가지 관점만 볼 수 있음
- 8개의 head → 8가지 다른 관점에서 분석
- 예: 주어-동사 관계, 명사-형용사 관계 등을 동시에 학습

### 3. Feed Forward - "단순한 변환"

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # 확장했다가 다시 축소
        return self.w_2(F.relu(self.w_1(x)))
```

Attention으로 정보를 모았으면, 이제 그걸 처리하는 단계.

### 4. Positional Encoding - "순서 정보"

Transformer는 병렬 처리라서 순서를 모르기 때문에 각 단어의 위치 정보를 따로 넣어줘야 한다.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # sin, cos 함수로 위치마다 다른 값 생성
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 단어 embedding에 위치 정보를 더함
        return x + self.pe[:, :x.size(1)]
```

## Encoder와 Decoder의 차이

### Encoder
```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask):
        # 1. Self-Attention (자기 자신 내에서 attention)
        x = x + self.self_attn(x, x, x, mask)
        # 2. Feed Forward
        x = x + self.feed_forward(x)
        return x
```

입력 문장을 이해하는 부분.

### Decoder
```python
class DecoderLayer(nn.Module):
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. Masked Self-Attention (앞 단어들만 볼 수 있음)
        x = x + self.self_attn(x, x, x, tgt_mask)
        # 2. Cross-Attention (Encoder 정보 참고)
        x = x + self.cross_attn(x, encoder_output, encoder_output, src_mask)
        # 3. Feed Forward
        x = x + self.feed_forward(x)
        return x
```

**핵심 차이**: Decoder는 미래를 볼 수 없다 (Masked Attention)
- 번역할 때 다음 단어를 미리 보면 안 되니까!

## 간단한 예제

```python
# 모델 생성
model = make_model(src_vocab=1000, tgt_vocab=1000, N=6)

# 학습
for epoch in range(100):
    # 입력: "I love you" → 출력: "Je t'aime"
    src = encode_sentence("I love you")
    tgt = encode_sentence("Je t'aime")
    
    output = model(src, tgt, src_mask, tgt_mask)
    loss = criterion(output, tgt)
    
    loss.backward()
    optimizer.step()

# 추론
src = encode_sentence("I love you")
translation = model.translate(src)
# → "Je t'aime"
```


## 요약

- **Encoder**: 입력 문장 이해 (Self-Attention × N)
- **Decoder**: 번역 생성 (Masked Self-Attention + Cross-Attention × N)
- **핵심**: Attention으로 문장 전체를 동시에 보면서 중요한 부분에 집중


## 참고
- 논문: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- 코드: [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
