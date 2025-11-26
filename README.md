# ResNet PyTorch 구현 분석

ResNet 논문을 읽고 PyTorch로 직접 구현하면서 정리한 내용입니다. 특히 Bottleneck 구조와 shortcut connection 처리가 헷갈려서 코드를 뜯어보며 정리했습니다.

## 기본 구조

ResNet은 크게 4개의 stage로 나뉘고, 각 stage는 여러 개의 residual block으로 구성됩니다.

```
Conv1 (7x7) → MaxPool → [Stage 1] → [Stage 2] → [Stage 3] → [Stage 4] → AvgPool → FC
```

## Residual Block의 두 가지 타입

### 1. BasicBlock (ResNet-18, 34에 사용)

```python
class Block(nn.Module):
    expansion = 1  # 채널 수 변화 없음
    
    def forward(self, x):
        identity = x
        
        # 3x3 conv → BN → ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv → BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # shortcut connection
        out += identity
        out = self.relu(out)
        return out
```

두 개의 3x3 conv로만 구성된 간단한 구조입니다.

### 2. Bottleneck (ResNet-50, 101, 152에 사용)

```python
class Bottleneck(nn.Module):
    expansion = 4  # 출력 채널이 4배로 늘어남
    
    def forward(self, x):
        identity = x
        
        # 1x1 conv: 채널 축소 (64 → 64)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv: 실제 feature 추출 (64 → 64)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1 conv: 채널 복원 및 확장 (64 → 256)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # shortcut connection
        out += identity
        out = self.relu(out)
        return out
```

**왜 Bottleneck을 쓸까?**
- 3x3 conv를 2개 쌓는 대신 1x1 → 3x3 → 1x1로 구성
- 중간 채널을 줄여서 연산량을 대폭 감소시킴
- 깊은 네트워크에서 필수적

## 핵심: Shortcut Connection과 차원 맞추기

ResNet의 핵심 아이디어는 `y = F(x) + x` 입니다. 

```python
out += identity  # 이 한 줄이 ResNet의 핵심
```

하지만 문제가 있습니다. F(x)와 x의 shape이 다를 수 있다는 점이죠.

### 언제 shape이 달라질까?

1. **Spatial dimension 변화**: stride=2로 feature map 크기가 줄어들 때
2. **Channel 변화**: expansion으로 채널 수가 늘어날 때

이럴 때 `i_downsample`을 사용해서 identity의 shape을 맞춰줍니다.

```python
# _make_layer 함수 내부
if stride != 1 or self.in_channels != planes * ResBlock.expansion:
    i_downsample = nn.Sequential(
        nn.Conv2d(self.in_channels, 
                  planes * ResBlock.expansion,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.BatchNorm2d(planes * ResBlock.expansion)
    )
```

1x1 conv를 사용해서 identity를 projection시킵니다. 이를 **projection shortcut**이라고 부릅니다.

## _make_layer 함수 이해하기

이 함수가 각 stage를 만드는 핵심 함수입니다.

```python
def _make_layer(self, ResBlock, blocks, planes, stride=1):
    layers = []
    
    # 첫 번째 블록: downsampling 발생 가능
    layers.append(ResBlock(self.in_channels, planes, 
                          i_downsample=i_downsample, stride=stride))
    
    # 나머지 블록들: 같은 차원 유지
    for i in range(blocks - 1):
        layers.append(ResBlock(self.in_channels, planes))
    
    return nn.Sequential(*layers)
```

예를 들어 ResNet-50의 `[3, 4, 6, 3]`은:
- Stage 1: Bottleneck 3개
- Stage 2: Bottleneck 4개  
- Stage 3: Bottleneck 6개
- Stage 4: Bottleneck 3개

각 stage의 첫 번째 블록에서만 downsampling이 일어납니다.

## 헷갈렸던 부분들

### Q1. expansion이 뭐지?
A: Bottleneck에서 최종 출력 채널이 중간 채널의 몇 배인지 나타냅니다.
- BasicBlock: expansion = 1 (64 → 64 → 64)
- Bottleneck: expansion = 4 (64 → 64 → 256)

### Q2. planes는 뭐고 out_channels는 뭐지?
A: 
- `planes`: 중간 채널 수 (예: 64)
- 실제 출력 채널: `planes * expansion` (예: 64 * 4 = 256)

### Q3. 왜 첫 번째 블록만 stride=2를 적용하나?
A: 한 stage 내에서 한 번만 downsampling 하면 되니까. 나머지는 같은 크기 유지.

## 전체 구조 예시 (ResNet-50)

```
Input (3, 224, 224)
↓
Conv1 + MaxPool → (64, 56, 56)
↓
Stage 1 (3 blocks) → (256, 56, 56)   # stride=1, no downsample
↓
Stage 2 (4 blocks) → (512, 28, 28)   # stride=2, downsample
↓
Stage 3 (6 blocks) → (1024, 14, 14)  # stride=2, downsample
↓
Stage 4 (3 blocks) → (2048, 7, 7)    # stride=2, downsample
↓
AvgPool + FC → (num_classes)
```

## 참고

- 논문: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- 코드 구현하면서 가장 도움 된 부분: i_downsample의 필요성과 _make_layer의 로직
