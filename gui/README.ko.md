[中文](./README.md) | [English](./README.en.md) | [日本語](./README.ja.md) | [한국어](./README.ko.md)

# Musubi Tuner GUI

musubi-tuner의 전체 워크플로우를 관리하는 NiceGUI 기반 그래픽 인터페이스.

## 기능

- 🎨 **전체 워크플로우 지원**: 데이터셋 태깅 → 캐시 → 학습 → 추론
- 🤖 **다중 아키텍처 지원**: FLUX.2, Wan2.1, HunyuanVideo, FramePack, Long-CAT, Z-Image, Qwen Image, HV 1.5, Lens, Ideogram-4, HiDream O1, FLUX Kontext, Krea-2 등
- 💾 **프리셋 관리**: 자주 사용하는 설정 저장 및 불러오기
- 📝 **실시간 로그**: 명령 출력 및 진행 상황 확인
- 🌐 **크로스 플랫폼**: Windows/Linux 지원, 로컬 또는 클라우드에서 실행 가능
- ⚡ **직접 호출**: PowerShell에 의존하지 않고 Python 스크립트를 직접 호출
- 🌓 **테마 전환**: 다크/라이트 테마 지원, 설정 자동 저장
- 🌐 **국제화 (i18n)**: 중국어, 영어, 일본어, 한국어 4개 언어 지원
- 🧪 **고급 학습**: SOAR 보조 학습, D-OPSD 증류 등 지원

## 설치

```bash
# 프로젝트 디렉토리로 이동
cd musubi-tuner-scripts

# 프로젝트 및 GUI 의존성 설치
uv sync --extra cu130 --extra gui --extra lycoris --extra attention --index-strategy unsafe-best-match

# musubi-tuner의 모든 의존성(torch, accelerate 등)이 설치되어 있는지 확인
```

## 사용 방법

### 방법 1: 루트 디렉토리 실행 스크립트

```powershell
# 프로젝트 루트에서 실행
./1.6.GUI.ps1

# 포트 지정
./1.6.GUI.ps1 -Port 8888

# 클라우드 모드 (외부 접근 허용)
./1.6.GUI.ps1 -Cloud

# 네이티브 창 모드
./1.6.GUI.ps1 -Native

# 브라우저 자동 열지 않기
./1.6.GUI.ps1 -NoBrowser
```

### 방법 2: Python 직접 실행

```bash
# 프로젝트 루트에서 실행
python gui/launch.py

# 클라우드 모드 (외부 접근 허용)
python gui/launch.py --cloud

# 포트 지정
python gui/launch.py --port 8888

# 네이티브 창 모드
python gui/launch.py --native

# 브라우저 자동 열지 않기
python gui/launch.py --no-browser
```

### 방법 3: gui 디렉토리에서 실행

```bash
cd gui
python launch.py
```

## 워크플로우

1. **데이터셋 태깅** (`/tagging`)
   - Qwen-VL 등 태깅 모델 지원
   - 이미지 일괄 처리
   - 프롬프트 접두사/접미사 커스터마이즈

2. **캐시 처리** (`/cache`)
   - 모델 아키텍처 선택
   - 모델 경로 설정
   - Latent 및 Text Encoder 출력 사전 계산
   - `python -m musubi_tuner.xxx_cache_latents` 직접 호출

3. **LoRA 학습** (`/train`)
   - 멀티 탭으로 파라미터 정리
   - 기본 설정, 모델 경로, 학습 파라미터, 네트워크 구조, 옵티마이저, 고급 옵션
   - 학습 로그 실시간 표시
   - 프리셋 저장/불러오기 지원
   - `python -m accelerate.commands.launch musubi_tuner.xxx_train_network` 직접 호출

4. **추론 / 생성** (`/generate`)
   - 학습된 LoRA 가중치 사용
   - 생성 파라미터 조정
   - 참조 이미지 편집 지원
   - `python -m musubi_tuner.xxx_generate` 직접 호출

## 호출 방식

GUI는 PowerShell 스크립트에 의존하지 않고 Python 모듈을 직접 호출합니다:

```bash
# Latent 캐시
python -m musubi_tuner.flux_2_cache_latents --dataset_config=... --vae=...

# Text Encoder 캐시
python -m musubi_tuner.flux_2_cache_text_encoder_outputs --dataset_config=... --text_encoder=...

# 학습 (accelerate 사용)
python -m accelerate.commands.launch --mixed_precision=bf16 musubi_tuner.flux_2_train_network --dit=... --vae=...

# 추론
python -m musubi_tuner.flux_2_generate_image --dit=... --prompt=...
```

## 지원 모델 아키텍처

| 아키텍처 | 캐시 모듈 | 학습 모듈 | 생성 모듈 |
|------|---------|---------|---------|
| FLUX.2 | flux_2_cache_latents | flux_2_train_network | flux_2_generate_image |
| FLUX Kontext | flux_kontext_cache_latents | flux_kontext_train_network | flux_kontext_generate_image |
| Wan2.1 | wan_cache_latents | wan_train_network | wan_generate_video |
| HunyuanVideo | cache_latents | hv_train_network | hv_generate_video |
| FramePack | fpack_cache_latents | fpack_train_network | fpack_generate_video |
| Long-CAT | longcat_cache_latents | longcat_train_network | - |
| Z-Image | zimage_cache_latents | zimage_train_network | zimage_generate_image |
| HV 1.5 | hv_1_5_cache_latents | hv_1_5_train_network | hv_1_5_generate_video |
| Qwen Image | qwen_image_cache_latents | qwen_image_train_network | qwen_image_generate |
| Lens | lens_cache_latents | lens_train_network | lens_generate_image |
| Ideogram-4 | ideogram4_cache_latents | ideogram4_train_network | ideogram4_generate_image |
| HiDream O1 | hidream_o1_cache_pixel | hidream_o1_train_network | hidream_o1_generate_image |
| Krea-2 | krea2_cache_latents | krea2_train_network | krea2_generate_image |

## 프리셋

`gui/presets/` 디렉토리는 단계별로 하위 디렉토리로 구성되며, 각 디렉토리에 TOML 프리셋 파일이 포함되어 있습니다:

### `presets/cache/` - 캐시 프리셋

flux2, flux_kontext, framepack, hidream_o1, hunyuan_video, hv_1_5, ideogram4, krea2, lens, long_cat, qwen_image, wan2_1, zimage, zimage_dopsd

### `presets/train/` - 학습 프리셋

flux2, flux_kontext, framepack, hidream_o1, hidream_o1_dev, hunyuan_video, hv_1_5, ideogram4, krea2, lens, lens_finetune, lens_finetune_low_vram, lens_low_vram, long_cat, qwen_image, qwen_image_finetune, wan2_1, zimage, zimage_dopsd, zimage_dopsd_finetune, zimage_finetune

### `presets/generate/` - 생성 프리셋

flux2, flux_kontext, framepack, hidream_o1, hidream_o1_dev_edit_flow, hidream_o1_dev_flash, hunyuan_video, hv_1_5, ideogram4, krea2, lens, long_cat, qwen_image, wan2_1, zimage

### `presets/user/` - 사용자 커스텀 프리셋

GUI에서 저장한 커스텀 프리셋은 이 디렉토리에 저장됩니다.

## 프로젝트 구조

```
gui/
├── main.py              # 메인 진입점
├── launch.py            # 실행 스크립트
├── README.md            # 중국어 문서
├── README.en.md         # 영어 문서
├── README.ko.md         # 한국어 문서 (이 파일)
├── PARAMETERS.md        # 파라미터 매핑 문서
├── UPDATES.md           # 업데이트 노트
├── theme.py             # 테마 시스템 (sd-scripts 스타일 통합)
├── STYLES_REUSE.md      # 스타일 재사용 가이드
├── components/          # 재사용 가능 컴포넌트
│   ├── path_selector.py    # 경로 선택기
│   ├── log_viewer.py       # 로그 뷰어
│   ├── preset_manager.py   # 프리셋 관리자
│   ├── model_selector.py   # 모델 선택기
│   └── side_tools.py       # 사이드 툴바
├── wizard/             # 위저드 단계
│   ├── step0_setup.py      # 환경 확인
│   ├── step1_tagging.py    # 데이터셋 태깅
│   ├── step2_cache.py      # 캐시 처리
│   ├── step3_train.py      # 학습
│   ├── step4_generate.py   # 추론 / 생성
│   ├── step7_settings.py   # 설정 페이지
│   └── console_page.py     # 콘솔 페이지
├── utils/              # 유틸리티
│   ├── config_manager.py   # 설정 관리
│   ├── process_runner.py   # 프로세스 실행 (Python 직접 호출)
│   ├── model_catalog.py    # 모델 아키텍처 카탈로그
│   ├── port_utils.py       # 포트 해석
│   └── i18n.py             # 국제화 (sd-scripts에서 재사용)
├── presets/            # 프리셋 설정
│   ├── cache/              # 캐시 프리셋 (*.toml)
│   ├── train/              # 학습 프리셋 (*.toml)
│   ├── generate/           # 생성 프리셋 (*.toml)
│   └── user/               # 사용자 커스텀 프리셋
└── examples/           # 사용 예제
    └── reuse_styles_example.py
```

## 주의사항

1. **작업 디렉토리**: GUI는 기본적으로 프로젝트 루트에서 스크립트를 실행합니다. 경로가 올바르게 설정되어 있는지 확인하세요
2. **의존성**: musubi-tuner의 모든 의존성(torch, accelerate 등)이 설치되어 있어야 합니다
3. **VRAM**: 모델과 설정에 따라 상당한 VRAM이 필요할 수 있습니다
4. **프리셋**: 프리셋은 파라미터만 저장하며 모델 경로는 저장하지 않습니다. 실제 환경에 맞게 조정하세요
5. **Python 모듈**: `musubi-tuner` 디렉토리가 Python 경로에 있거나 패키지로 설치되어 있는지 확인하세요

## FAQ

**Q: 새로운 모델 아키텍처 지원을 추가하려면?**
A: `gui/utils/model_catalog.py`를 편집하여 `MODEL_CATALOG` 딕셔너리에 새 아키텍처를 추가하세요.

**Q: 학습 스크립트 파라미터를 커스터마이즈하려면?**
A: 해당 단계 페이지 코드(예: `step3_train.py`)에서 파라미터 구성 로직을 수정하세요.

**Q: 클라우드 서버에서 사용하려면?**
A: `--cloud` 파라미터로 실행한 후, 브라우저에서 `http://<서버IP>:7788`(기본 포트)에 접속하세요

**Q: 커스텀 설정을 저장하려면?**
A: 학습 페이지에서 "프리셋으로 저장" 버튼을 클릭하고 이름을 입력하세요.

**Q: 모듈을 찾을 수 없다는 에러가 나요?**
A: 프로젝트 루트에서 실행하고 `musubi-tuner` 디렉토리가 있는지 확인하세요. 다음을 시도해 볼 수 있습니다:
```python
import sys
sys.path.insert(0, '.')
```

## 테마 및 국제화

### 다크/라이트 테마

GUI는 다크와 라이트 테마를 모두 지원합니다. 우측 상단의 태양/달 아이콘을 클릭하여 전환합니다. 테마 설정은 브라우저의 `localStorage`에 저장되며, 다음 방문 시 자동으로 로드됩니다.

### Modern Theme (기본값)
- 다크 배경, 모던한 카드와 버튼
- 딥 그린 + 골드 내추럴 컬러 스킴

### Green Gold Theme (sd-scripts에서)
- 밝은 배경, 전통적인 그린 골드 컬러 스킴
- `sd-scripts/gui/styles.py`에서 재사용

### 국제화 (i18n)

4개 언어(중국어, 영어, 일본어, 한국어)를 지원합니다. 우측 상단의 언어 드롭다운을 클릭하여 전환합니다. 전환 후 페이지가 자동으로 새로고침되어 새 언어가 적용됩니다.

자세한 내용은 `STYLES_REUSE.md`, `UPDATES.md`, `examples/reuse_styles_example.py`를 참조하세요.

## 라이선스

musubi-tuner 메인 프로젝트와 동일합니다.
