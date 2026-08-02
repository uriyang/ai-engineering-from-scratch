# 수치적 안정성

수치적 안정성은 계산이 너무 커지거나 너무 작아져서 망가지지 않게 만드는 방법이다. 딥러닝에서는 loss가 `NaN`이 되거나, gradient가 0이 되거나, float 오차가 쌓여서 학습 결과가 달라지는 일이 자주 생긴다.

## IEEE 754: 부동소수점

- 컴퓨터는 실수를 정확히 저장하지 못하고 근사값으로 저장한다
- float에는 부호(sign), 지수(exponent), 가수(mantissa)가 있다
- exponent는 범위, mantissa는 정밀도를 결정한다
- float16은 범위가 좁고, bfloat16은 범위가 넓다

쉽게 말하면:
- 부동소수점은 “정확한 실수”가 아니라 “가까운 근사값”이다

## 0.1 + 0.2가 0.3이 아닌 이유

- 0.1은 이진수로 정확히 표현되지 않는다
- 이진수에서는 0.1이 반복되는 무한소수처럼 된다
- 그래서 컴퓨터는 0.1을 가까운 값으로 잘라서 저장한다
- 그래서 아주 작은 반올림 오차가 생긴다
- 이런 오차는 누적되면 눈에 띄는 차이가 된다

```text
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

실무에서는:
- float 비교는 `==`보다 epsilon을 두고 비교하는 편이 안전하다

## overflow와 underflow

- overflow는 너무 큰 값이 `inf`가 되는 현상이다
- underflow는 너무 작은 값이 `0.0`으로 사라지는 현상이다
- `exp()`와 `log()`는 특히 이 문제를 자주 일으킨다

예를 들어:
- 큰 logits에 `exp()`를 바로 적용하면 overflow가 난다
- 아주 작은 값에 `log()`를 취하면 `-inf`가 나올 수 있다

## catastrophic cancellation (치명적 상쇄)

- 거의 같은 두 수를 빼면 중요한 자리수가 사라진다
- 남는 값은 반올림 오차의 영향을 크게 받는다
- 분산 계산이나 로그 확률 차이에서 자주 생긴다

즉, 식을 그대로 쓰기보다 수치적으로 더 안전한 형태로 바꾸는 게 중요하다.

## log-sum-exp (로그-합-지수)

- `log(sum(exp(x)))`는 직접 계산하면 위험하다
- 가장 큰 값만 빼고 계산하면 overflow를 피할 수 있다
- 마지막에 다시 그 값을 더하면 원래 결과와 같아진다

```text
log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
```

이 트릭은:
- 소프트맥스 정규화
- 교차 엔트로피 손실 계산
- 순차 모델에서의 로그 확률 합산
- 가우스 분포의 혼합
- 변분 추론

같은 곳에서 자주 쓰인다.

## 안정한 softmax (최대값 빼기)

- softmax는 logits를 확률로 바꾼다
- logits가 너무 크면 `exp()`가 overflow할 수 있다
- max를 빼고 계산하면 같은 결과를 더 안전하게 얻을 수 있다

즉, 안정한 softmax는 선택이 아니라 사실상 필수다.

## NaN과 Inf

- `inf`는 무한대, `nan`은 숫자가 아님을 뜻한다
- 한 번 생기면 계산 전체로 퍼지기 쉽다
- `exp`, 0으로 나누기, `inf - inf`, `log(음수)`가 대표적인 원인이다

실제로는 이렇게 자주 나타난다:

- `inf`
  - `exp()`가 너무 큰 양수에 적용될 때
  - `1.0 / 0.0`
  - `float32` 범위를 넘는 큰 수
- `nan`
  - `0.0 / 0.0`
  - `inf - inf`
  - `inf * 0`
  - `sqrt()`의 음수
  - `log()`의 음수
  - 기존 `nan`이 섞인 모든 연산

예방 방법:
- `exp()` 입력을 clamp한다
- 분모에 작은 `epsilon`을 더한다
- `log(x)` 대신 `log(x + 1e-8)`처럼 epsilon을 넣는다
- 안정한 구현을 쓴다
- gradient clipping으로 가중치 폭주를 막는다
- forward pass마다 `nan`과 `inf`를 점검한다

## gradient checking (수치적 기울기 검사)

- 수학적으로 구한 gradient가 맞는지 확인할 때 쓴다
- 중심차분으로 수치 미분을 구해서 비교한다
- 너무 큰 step은 부정확하고, 너무 작은 step은 반올림 오차에 취약하다

중심 차분 공식:

```text
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

- `h`는 아주 작은 값이다
- 양쪽을 대칭으로 보기 때문에 더 안정적이다
- 새 레이어나 loss를 만들었을 때 analytical gradient와 비교해 확인한다

실무에서는:
- 새 레이어나 loss를 만들면 gradient check를 해보는 게 좋다

## mixed precision (혼합 정밀도 훈련)

- float16은 빠르지만 범위가 좁다
- bfloat16은 정밀도는 조금 낮아도 범위가 넓다
- 학습에서는 range가 더 중요해서 bfloat16이 유리한 경우가 많다

mixed precision은 보통:
- 가중치는 float32로 유지하고
- forward/backward는 더 낮은 정밀도로 계산한다

## loss scaling

- float16에서는 gradient가 너무 작아 0으로 사라질 수 있다
- loss에 큰 값을 곱해 gradient를 키운 뒤
- 업데이트 직전에 다시 나누면 underflow를 줄일 수 있다

즉, loss scaling은 작은 gradient를 살려주는 방법이다.

## gradient clipping

- gradient가 너무 커지면 한 번의 업데이트로 학습이 망가질 수 있다
- clip by value는 각 원소를 자른다
- clip by norm은 전체 크기를 제한한다

clip by value:

```text
grad = clamp(grad, -max_val, max_val)
```

- 각 gradient 값을 `-max_val`과 `max_val` 사이로 잘라낸다

clip by norm:

```text
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

- gradient 전체의 크기가 너무 크면 비율만 줄여서 방향은 유지한다

보통은:
- norm clipping이 더 자연스럽고 많이 쓰인다

## normalization

- batch norm, layer norm, RMS norm은 값을 안정적인 범위로 유지해준다
- 너무 커지거나 너무 작아지는 activation을 막는다
- forward와 backward 둘 다 안정화하는 데 도움이 된다

## 자주 나는 오류

- loss가 `NaN`이 된다
- loss가 `log(num_classes)` 근처에서 멈춘다
- `exp()`가 `inf`를 반환한다
- float32에서 float16으로 바꾼 뒤 학습이 발산한다
- 일부 레이어의 gradient norm이 0.0으로 나온다
- GPU마다 결과가 조금씩 다르다
- 검증 정확도가 예상보다 1~3% 낮다

이런 문제들은 대부분:
- 안정한 수식으로 바꾸기
- float 정밀도 조정
- clipping과 normalization 사용

으로 해결할 수 있다.
