# Transformer Learning Studio

트랜스포머의 내부 연산부터 공개 연구 기반 LLM 제작 공정까지 실제로 실행하고
시각화하는 로컬 교육용 웹앱입니다. UI와 수식 해설은 한국어, 모델 학습과 생성
텍스트는 영어로 제공됩니다.

## 두 개의 학습 트랙

### 학습 현미경

- 927,232 파라미터 decoder-only Transformer
- TinyStories 5,000개 학습·500개 검증 고정 부분집합
- 토큰화, embedding, Q/K/V, causal attention, FFN, logits 시각화
- loss, perplexity, gradient norm, 파라미터 변화량과 자기회귀 생성 trace
- CPU 실행 지원 및 준비된 600-step 체크포인트 포함

### 모델 공장

- 15,538,560 파라미터 `ProductionMiniLM`
- BPE 8,192, context 128, dimension 384, 8 layers
- 8 query heads / 2 KV heads GQA, RMSNorm, RoPE, SwiGLU, tied embedding
- TinyStories 중복·품질·언어·반복 필터와 PII 마스킹, sequence packing
- `base → SFT → {DPO, PPO, GRPO}` 체크포인트 흐름
- Bradley-Terry 보상모델과 LoRA rank 8 사후학습
- BF16/FP16 mixed precision, gradient clipping, OOM microbatch 재시도
- 최근 공장 실행 3개와 각 실행의 safetensors 체크포인트 보존

상용 기업의 비공개 학습법을 재현한다고 주장하지 않습니다. 공개 논문의 대표
공정을 RTX 4060에서도 관찰할 수 있는 교육용 크기로 구현한 것입니다.

## 화면

- **모델 공장**: 데이터 정제부터 양자화까지 단계별 또는 자동 실행
- **뉴럴넷 관측실**: activation 흐름, 레이어별 역전파, AdamW 파라미터 이동,
  실제 5×5 local loss landscape
- **학습 현미경**: 기존 가이드 학습과 자유 실험실
- **사후학습**: SFT, 보상모델, PPO, DPO, GRPO 수식과 실제 지표
- **피드백 아레나**: 출처를 숨긴 A/B 선택과 사용자 선호 저장
- **추론 엔진**: prefill/decode, KV cache, MHA/GQA/MQA, continuous batching
- **평가·비교**: 정렬 모델 지표, reward hacking, BF16/INT8/INT4 비교

실제 GPU 연산과 7B·70B 규모 추정 시뮬레이션은 화면 배지로 구분됩니다.

## GPU 설치와 실행

Windows에서 `start.bat`을 더블클릭하면 됩니다. 첫 실행에는 Node.js 20 이상,
Python 3.11 이상과 최신 NVIDIA 드라이버가 필요합니다.

```powershell
.\start.bat
```

런처는 프런트엔드 패키지와 빌드 결과를 확인하고, NVIDIA GPU가 있으면 프로젝트
전용 `.venv-gpu`와 PyTorch 2.11.0 CUDA 13.0 환경을 자동으로 준비합니다.
첫 설치는 큰 PyTorch 패키지를 내려받으므로 몇 분 걸릴 수 있습니다. GPU 설치가
실패하거나 NVIDIA GPU가 없으면 학습 현미경용 `.venv` CPU 환경으로 전환합니다.
서버 준비가 끝나면 `http://127.0.0.1:8000`을 자동으로 엽니다.

수동으로 GPU 환경만 먼저 준비하려면 다음 명령을 사용할 수 있습니다.

```powershell
.\setup_gpu.ps1
```

실행이 실패하면 창에 원인이 표시되고 프로젝트 루트의 `startup.log`에도
진단 내용이 남습니다.

CUDA가 준비되지 않으면 모델 공장 실행은 차단되고 GPU, 드라이버, PyTorch
빌드, VRAM과 설치 명령을 `/api/system` 및 UI에서 진단합니다. 학습 현미경은
일반 Python 환경에서 CPU로 계속 사용할 수 있습니다.

## 기본 공정

기본값은 effective batch 32를 `microbatch 16 × accumulation 2`로 구성합니다.
OOM이 발생하면 microbatch를 절반으로 줄이고 accumulation을 두 배로 늘려 한
번 재시도합니다.

| 단계 | 기본 업데이트 |
| --- | ---: |
| Pretraining | 1,200 |
| SFT | 200 |
| Reward model | 150 |
| DPO | 100 |
| PPO | 30 |
| GRPO | 30 |

RTX 4060에서 기본 전체 공정을 직접 실행한 결과 5분 14초가 걸렸고, peak
VRAM은 약 1.12GB였습니다. 그중 1,200-step pretraining은 약 1분 53초가
걸렸습니다. 환경과 동시 실행 프로그램에 따라 전체 시간은 달라질 수 있습니다.

## KV cache와 양자화

forward는 layer별 `past_key_values`를 받고 prefill과 one-token decode를
분리합니다. 캐시는 `[batch, kv_heads, sequence, head_dim]`으로 표시되며
greedy cached/uncached 출력 일치를 검증합니다.

캐시 메모리는 다음 식으로 비교합니다.

```text
2 × layers × kv_heads × head_dim × tokens × bytes
```

INT8은 실제 per-channel, INT4는 실제 groupwise quantize/dequantize 오차를
계산합니다. fused quantized kernel은 구현하지 않았으므로 속도 향상을 보장하지
않습니다. 작은 eager 모델과 짧은 문맥에서는 동적 cache 연결 비용 때문에
cached 생성이 더 느릴 수도 있습니다.

## 테스트

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q
cd frontend
cmd /c npm test -- --run
cmd /c npm run build
```

테스트는 RoPE, RMSNorm, SwiGLU, GQA, cache logits 일치, LoRA device, SFT
masking, reward pair loss, PPO clipping, DPO, GRPO, 사용자 투표, 양자화,
공장 보존 정책과 API 계약을 포함합니다.

## 주요 API

- `GET /api/system`
- `POST /api/factory/runs`
- `POST /api/factory/runs/{id}/control`
- `WS /ws/factory/{id}`
- `GET /api/preferences/next`
- `POST /api/preferences`
- `POST /api/inference/kv-benchmark`
- `POST /api/scale/estimate`

기존 `/api/runs`, `/api/generate`, `/api/inspect`, `/ws/runs/{id}`도 유지됩니다.

## 데이터와 출처

전체 TinyStories를 포함하지 않고 교육용 고정 부분집합만 저장합니다. 최초 설치
이후 데이터 탐색, 체크포인트 추론과 재학습은 오프라인으로 동작합니다.

- 데이터: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- TinyStories 논문: [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- 데이터 라이선스: **CDLA-Sharing-1.0**
- PPO/RLHF: [InstructGPT](https://arxiv.org/abs/2203.02155)
- DPO: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- GRPO: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- KV cache: [Hugging Face cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation)
