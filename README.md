# ResNet(Residual Network) 아키텍처의 PyTorch 구현(상세)

## 주요 구성 요소

1. **`Bottleneck` 클래스**: 
    
    더 깊은 ResNet 변형(ResNet50, 101, 152 등)에 사용되는 "병목(bottleneck)" 구조의 잔차 블록(residual block)을 구현 → *논문 그림5(오른쪽) 구조 참조*
    
    - `expansion=4` :출력 채널이 입력 `out_channels`(중간 계층의 채널 수)의 4배임을 의미
        - 그림 5를 보면 256(64*4)으로 채널이 확장됨
    - **`init**(self, ...)` :
        - 1x1, 3x3, 1x1 컨볼루션 레이어를 사용하여 차원을 줄인 다음 복원하여 계산 효율성을 높임
        - `self.conv1`:  1x1 컨볼루션으로, 입력 채널 수를 줄여 연산량을 감소시키는 ‘병목’ 역할
        - `self.conv2` : 3x3 컨볼루션으로, 줄어든 채널 상태에서 실제 공간적 특징(spatial feature)추출
        - `self.conv3` : 1x1 컨볼루션으로 채널수를 expansion배수만큼 복원하고 늘려줌
        - `self.i_downsample` : Shortcut 연결을 위한 부분. 만약 블록을 통과하면서 입력과 출력의 차원이 달라질 경우 ,downsample 레이어를 통해 입력(identity)의 차원을 출력과 동일하게 맞춤
    - **`forward(self, x)`**:
        - `identity = x.clone()`: Shortcut 연결을 위해 초기 입력값을 `identity` 변수에 복사
        - `x = self.relu(...)`: 입력 `x`가 `conv1`, `conv2`, `conv3`를 순차적으로 통과하며 연산됨. 각 컨볼루션 후에는 배치 정규화(Batch Normalization)와 ReLU 활성화 함수가 적용
        - `if self.i_downsample is not None:`: 만약 차원 축소가 필요하다면(`i_downsample`이 정의되었다면), `identity`에 프로젝션 Shortcut 연산을 적용
        - `x += identity`: **ResNet의 핵심.** 컨볼루션 계층들을 통과한 결과(mathcalF(x))에 초기 입력값(`identity`, 즉 x)을 더해줍니다. 이는 논문의 수식
            
            $y=mathcalF(x,W_i)+x$  에 해당
            
        - `x = self.relu(x)`: 잔차(residual)가 더해진 결과에 최종적으로 ReLU 활성화 함수 적용
    
2. **`Block` 클래스**: 
    
    더 얕은 ResNet 변형(ResNet18, 34 등)에 사용되는 기본 블록을 구현 → *논문 그림5(왼쪽) 구조 참조*
    
    - **`expansion = 1`**: 기본 블록에서는 채널 수가 확장되지 않기 때문에 1
    - **`init**(self, ...)`: 두 개의 3x3 컨볼루션 레이어로 구성
    - **`forward(self, x)`**: Bottleneck 클래스와 원리는 동일, 3x3컨볼루션 두 개를 통과한 결과에 identity 를 더하는 더 단순한 구조
        - 배치 정규화 및 ReLU 활성화로 구성
        - `Bottleneck`과 유사하게 잔여 연결도 처리
    
3. **`ResNet` 클래스**: 
    
    전체 ResNet 모델을 구축하는 주요 클래스 *→그림3, 표1 에 제시된 전체 네트워크 구조에 해당*
    
- **`__init__(self, ...)`**:
    - `self.conv1`, `self.max_pool`: 모든 ResNet 모델의 시작 부분에 공통으로 존재하는 초기 컨볼루션 레이어와 맥스 풀링 레이어
        - 초기 컨볼루션 레이어, 배치 정규화, ReLU 및 최대 풀링으로 시작
    - `self.layer1` ~ `self.layer4`: `_make_layer` 헬퍼 함수를 호출하여 각 스테이지를 구성
        - 각 레이어는 여러 잔여 블록으로 구성되며 공간 차원을 줄이기 위한 다운샘플링(`stride=2`)을 포함
    - `self.avgpool`, `self.fc`: 최종적으로 특징 맵을 1x1 크기로 줄이는 전역 평균 풀링 (Global average Pooling) 레이어와 클래스를 분류하기 위한 완전 연결 레이어(Fully Connected Layer)
- `forward(self, x)`: 네트워크의 순방향 전달을 정의
- **`_make_layer(self, ...)`**: 특정 수의 잔차 블록으로 구성된 하나의 스테이지를 만드는 역할
    - `if stride != 1 or self.in_channels != …` : 스테이지의 첫 블록에서 다운샘플링이 필요한지 확인하는 핵심 로직. 
    스테이지가 바뀌면서 채널 수가 변하거나, 공간적 크기를 절반으로 줄여야 할 때(`stride=2`), `i_downsample` (프로젝션 Shortcut)을 생성
    - `for i in range(blocks-1)`: 첫 블록을 제외한 나머지 블록들은 입력과 출력의 차원이 같으므로 `i_downsample` 없이 생성
    - `return nn.Sequential(*layers)`: 생성된 블록들을 하나의 `nn.Sequential` 모듈로 묶어 반환
    
1. **ResNet 함수(`ResNet50`, `ResNet101`, `ResNet152`)**: 
    
     특정 ResNet 모델을 쉽게 생성하기 위한 **팩토리 함수(Factory Functions)**
    
    - 이 함수는 적절한 블록 유형(`Bottleneck`)과 각 레이어에 대한 블록 목록을 `ResNet` 클래스 생성자에 전달하여 특정 ResNet 모델을 생성하는 팩토리 메서드
    - 표 1 에 명시된 각 모델의 구성에 맞게 ResNet 클래스를 호출
        - 예) 숫자의 의미: ResNet-50의 [3, 4, 6, 3]
        
        ResNet은 여러 스테이지(stage)로 구성됩니다. `conv1` 레이어 이후 `conv2_x`, `conv3_x`, `conv4_x`, `conv5_x`의 4개 스테이지가 있습니다.
        
        `[3, 4, 6, 3]`은 각 스테이지에 몇 개의 **Bottleneck 블록**을 쌓을지를 나타냅니다.
        
        - `conv2_x`: **3**개의 Bottleneck 블록
        - `conv3_x`: **4**개의 Bottleneck 블록
        - `conv4_x`: **6**개의 Bottleneck 블록
        - `conv5_x`: **3**개의 Bottleneck 블록
    

## 정리 및 주요 코드 확인

1. Shortcut 연결: 잔차 학습의 구현 (`x += identity`)

- 가장 중요한 부분은 `Bottleneck`과 `Block` 클래스의 `forward` 함수에 있는 `x += identity` 라인
- 논문의 핵심 아이디어인 **잔차 학습(Residual Learning)**을 코드로 구현한 것

2. 차원 맞춤: 유연한 연결을 위한 프로젝션 (`i_downsample`)

- Shortcut 연결에서 `$\mathcal{F}(x)$`와 x를 더하려면 두 텐서의 모양(shape)이 같아야 하나 네트워크가 깊어지면서 특징 맵(feature map)의 크기가 줄어들거나 채널 수가 늘어나는 경우가 생김
- 이때 `i_downsample`이 이 문제를 해결하는 중요한 역할.
- `_make_layer` 함수 안의 조건문은 바로 이 `i_downsample`이 언제 필요한지를 결정하는 부분
    - **`stride != 1`**: 블록의 첫 번째 컨볼루션에서 `stride`가 2로 설정되어 특징 맵의 너비와 높이가 절반으로 줄어들 때.
    - **`self.in_channels != planes*ResBlock.expansion`**: 블록을 통과하면서 채널의 개수가 변할 때.
    
    이 조건에 해당하면, `i_downsample`은 1x1 컨볼루션을 통해 원본 입력 `identity`의 크기와 채널 수를 $\mathcal{F}(x)$와 동일하게 맞춤. → 프로젝션 Shortcut(Projection Shortcut)
    

3. 네트워크 조립: 재사용 가능한 블록 구조 (`_make_layer`)

- `ResNet` 클래스의 `_make_layer` 함수는 ResNet 아키텍처의 모듈성과 확장성을 보여주는 중요한 부분
    - ResNet-18부터 ResNet-152까지, 심지어 1000개가 넘는 레이어의 네트워크도 동일한 기본 블록을 재사용하여 구성 가능
    - `ResBlock` (Block 또는 Bottleneck), `blocks` (블록 개수), `planes` (채널 수)를 인자로 받아, 첫 번째 블록에는 필요시 `i_downsample`을 적용하고, 나머지 블록들은 루프를 돌며 쌓아 하나의 `nn.Sequential` 모듈로 만들어 반환
    - 덕분에 `ResNet50([3, 4, 6, 3], ...)`처럼 간단한 호출만으로 복잡한 네트워크를 일관성 있게 생성할 수 있게됨
