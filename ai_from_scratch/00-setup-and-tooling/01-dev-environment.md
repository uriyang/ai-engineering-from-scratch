# CUDA
- NVIDIA 그래픽카드(GPU)를 이용해서 연산을 빠르게 처리할 때 사용

원래 GPU는 화면이나 3D 그래픽을 그리는 장치였는데, CUDA를 사용하면 개발자가 GPU를 대규모 병렬 계산 장치처럼 활용할 수 있어요.

대표적으로는:
- AI·딥러닝 학습/추론
PyTorch나 TensorFlow에서 모델을 GPU로 돌릴 때 사용해요.
- 영상·이미지 처리
영상 인코딩, 필터 적용, 이미지 분석처럼 같은 계산을 픽셀마다 반복할 때 유리해요.
- 과학·수치 계산
행렬 계산, 시뮬레이션, 물리·금융 모델링 등에 사용해요.
- 3D 렌더링
Blender 렌더링이나 그래픽 연산을 빠르게 할 때 활용해요.

예를 들어 PyTorch에서는 이렇게 사용해요.

```
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
data = data.to(device)
```

여기서 cuda는 “CPU 말고 NVIDIA GPU에서 계산해”라는 뜻이에요.

다만 CUDA는 아무 작업에나 빠른 것은 아니에요. 작은 계산, 조건문이 많은 작업, 파일 읽기 같은 작업은 CPU가 더 나을 수 있고, 큰 행렬이나 동일한 계산을 수천·수만 번 동시에 처리할 때 특히 효과가 커요.

그리고 CUDA는 기본적으로 NVIDIA GPU 전용이에요. AMD GPU는 ROCm, 애플 실리콘은 Metal/MPS 같은 기술을 사용해요.

# UV
- Python 개발 환경과 패키지를 `uv`라는 도구로 관리

| 환경      | 실행 기반   | 패키지·프로젝트 관리 |
| ------- | ------- | ----------- |
| Python  | Python  | `uv`        |
| Node.js | Node.js | `pnpm`      |

- uv를 사용하면 Python 설치, 가상환경, 패키지 설치, 버전 고정, 명령어 실행 등을 하나의 도구로 관리할 수 있어요. 공식 문서에서도 uv를 Rust로 작성된 빠른 Python 패키지·프로젝트 관리 도구라고 설명합니다.

| pnpm                | uv                   |
| ------------------- | -------------------- |
| `pnpm init`         | `uv init`            |
| `pnpm add axios`    | `uv add requests`    |
| `pnpm install`      | `uv sync`            |
| `pnpm remove axios` | `uv remove requests` |
| `pnpm exec ...`     | `uv run ...`         |
| `package.json`      | `pyproject.toml`     |
| `pnpm-lock.yaml`    | `uv.lock`            |
| `node_modules`      | `.venv`              |

# fnm VS nvm
- fnm: 빠르고 가벼운 Node 버전 관리자
- nvm: 가장 널리 알려진 전통적인 Node 버전 관리자

## 차이점:
- fnm은 Rust 기반이라 빠른 편
- nvm은 오래되어 문서/예제가 더 많음


# GPU 설정
- 이 Mac은 Apple M3 Pro 내장 GPU가 있어요.
- 그래서 nvidia-smi는 안 됩니다. NVIDIA GPU 전용 명령이라서요.
- PyTorch에서는 cuda가 아니라 mps를 써야 해요.
- 설치는 이렇게 하면 됩니다.
```bash
uv pip install torch torchvision torchaudio
```
- 확인은 이렇게 하면 됩니다.
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
