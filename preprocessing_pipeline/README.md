# EEG 전처리 파이프라인

## 개요

이 파이프라인은 감정 자극에 대한 EEG 데이터를 전처리하고 에포크를 생성하는 모듈화된 시스템입니다. 다양한 아티팩트 제거 방법(ICA, LMD-DWT, ICA-DWT Hybrid)을 지원하며, 체계적인 데이터 처리와 결과 저장을 제공합니다.

## 주요 기능

### 🧠 다중 아티팩트 제거 방법
- **ICA (Independent Component Analysis)**: 전통적인 독립성분분석
- **LMD-DWT**: Local Mean Decomposition + Discrete Wavelet Transform
- **ICA-DWT Hybrid**: ICA와 DWT를 결합한 하이브리드 방법

### 📊 체계적인 데이터 처리
- 자동 데이터 검증 및 로딩
- MNE 기반 전처리 파이프라인
- 에포크 생성 및 저장
- 처리 결과 요약 및 통계

### 🔧 모듈화된 구조
- 기능별 분리된 모듈들
- 재사용 가능한 컴포넌트
- 명확한 에러 처리 및 로깅

## 프로젝트 구조

```
preprocessing_pipeline/
├── eeg_preprocessing.py              # 메인 실행 스크립트
├── config.py                         # 설정 관리
├── data_loader.py                    # 데이터 로딩 및 검증
├── mne_processor.py                  # MNE 기반 전처리
├── ica_processor.py                  # ICA 아티팩트 제거
├── lmd_dwt_processor.py              # LMD-DWT 아티팩트 제거
├── ica_dwt_hybrid_processor.py       # ICA-DWT 하이브리드 제거
├── epoch_processor.py                # 에포크 생성 및 저장
├── marker_parser.py                  # 마커 파싱 및 이벤트 처리
├── analysis_utils.py                 # 분석 유틸리티 함수들
└── README.md                         # 이 파일
```

## 모듈 의존성 관계

### 📊 의존성 다이어그램

```
eeg_preprocessing.py (메인)
├── config.py
├── data_loader.py
├── mne_processor.py
│   └── marker_parser.py
├── ica_processor.py
├── lmd_dwt_processor.py
├── ica_dwt_hybrid_processor.py
├── epoch_processor.py
└── analysis_utils.py
    ├── data_loader.py
    └── marker_parser.py
```

### 🔗 상세 Import 관계

#### **1. eeg_preprocessing.py (메인 스크립트)**
```python
from config import SUBJECTS, DATA_DIR, OUTPUT_DIR, DEFAULT_TEST_MODE
from data_loader import find_subject_file, load_csv, validate_data_file
from mne_processor import create_mne_raw, preprocess_data
from ica_processor import process_with_ica
from lmd_dwt_processor import process_with_lmd_dwt
from ica_dwt_hybrid_processor import process_with_ica_dwt_hybrid
from epoch_processor import create_epochs, save_results
from analysis_utils import (
    analyze_event_counts, 
    create_processing_summary,
    validate_processing_pipeline
)
```

#### **2. mne_processor.py**
```python
from marker_parser import create_annotations_from_markers
```

#### **3. analysis_utils.py**
```python
from data_loader import find_subject_file, load_csv
from marker_parser import parse_marker, analyze_markers
```

## 사용 방법

### 1. 기본 실행

```bash
cd preprocessing_pipeline
python eeg_preprocessing.py
```

### 2. 클래스 기반 사용법

```python
from eeg_preprocessing import EEGPreprocessor
from config import PreprocessingConfig

# 기본 설정으로 전처리기 초기화
preprocessor = EEGPreprocessor()
preprocessor.run_processing()
```

### 3. 사용자 정의 설정

```python
from eeg_preprocessing import EEGPreprocessor
from config import PreprocessingConfig

# 사용자 정의 설정
config = PreprocessingConfig(
    subjects=['3', '4', '5', '6', '9', '10'],
    test_mode=False,
    artifact_method='ica_dwt_hybrid',
    data_dir='.../1_data',
    output_dir='.../2_output'
)

# 사용자 정의 설정으로 전처리기 초기화
preprocessor = EEGPreprocessor(config=config)
preprocessor.run_processing()
```

### 4. 개별 참가자 처리

```python
from eeg_preprocessing import EEGPreprocessor
from config import PreprocessingConfig

# 전처리기 초기화
config = PreprocessingConfig(artifact_method='ica')
preprocessor = EEGPreprocessor(config=config)

# 단일 참가자 처리
success = preprocessor.process_subject('3')
if success:
    print("참가자 3 처리 완료")
```

### 5. 기존 함수 호환성 (하위 호환성)

```python
from eeg_preprocessing import process_subject
from pathlib import Path

# 단일 참가자 처리 (기존 방식)
success = process_subject(
    subject="3",
    data_dir=Path(".../1_data"),
    output_dir=Path(".../2_output"),
    test_mode=False,
    artifact_method="ica"
)
```

### 6. 설정 기반 초기화

```python
from eeg_preprocessing import EEGPreprocessor
from config import PreprocessingConfig

# 설정 객체 생성
config = PreprocessingConfig()

# 설정 기반으로 전처리기 초기화
preprocessor = EEGPreprocessor(
    config=config,
    data_dir=config.data_dir,
    output_dir=config.output_dir,
    test_mode=config.test_mode,
    artifact_method=config.artifact_method
)

# 특정 참가자만 처리
preprocessor.run_processing(subjects=['3', '4', '5'])
```

## 출력 파일

### 1. Raw 데이터 파일
- `{subject}_raw.fif`: 전처리된 Raw 데이터
- `{subject}_filtered_raw.fif`: 필터링된 Raw 데이터

### 2. 에포크 파일
- `{subject}_epo.fif`: 생성된 에포크 데이터
- `{subject}_epo.fif.json`: 에포크 메타데이터

### 3. 시각화 파일
- `{subject}_{method}_comparison.png`: 전처리 전후 비교
- `{subject}_{method}_spectral.png`: 스펙트럼 밀도 비교

### 4. 처리 로그
- 콘솔 출력: 실시간 처리 상황
- 오류 메시지: 상세한 오류 정보

## 설정 옵션

### config.py에서 수정 가능한 설정

```python
# 참가자 ID 리스트
SUBJECTS = [3, 4, 5, 6, 7, 8, 9, 10, ...]

# 데이터 디렉토리
DATA_DIR = "../1_data"

# 출력 디렉토리
OUTPUT_DIR = "2_output"

# 기본 테스트 모드
DEFAULT_TEST_MODE = True
```

## 아티팩트 제거 방법 비교

### 1. ICA (Independent Component Analysis)
**장점:**
- 전통적이고 검증된 방법
- 시각적 성분 검토 가능
- 다양한 아티팩트 제거 가능

**단점:**
- 수동 검토 필요
- 처리 시간이 상대적으로 김

### 2. LMD-DWT (Local Mean Decomposition + DWT)
**장점:**
- 자동화된 처리
- 분산 엔트로피 기반 정확한 감지
- 다중 임계값 기반 판단

**단점:**
- 복잡한 알고리즘
- 계산 비용이 높음

### 3. ICA-DWT Hybrid
**장점:**
- 두 방법의 장점 결합
- 순차적 정리로 높은 정확도
- 자동화된 처리

**단점:**
- 가장 복잡한 방법
- 처리 시간이 가장 김

## 문제 해결

### 일반적인 오류 및 해결 방법

1. **데이터 파일을 찾을 수 없음**
   ```
   ❌ 참가자 X의 데이터 파일을 찾을 수 없습니다.
   ```
   **해결:** `config.py`의 `DATA_DIR` 경로 확인

2. **채널 수 불일치 오류**
   ```
   ❌ len(data) (4) does not match len(info["ch_names"]) (13)
   ```
   **해결:** LMD-DWT 처리 모듈이 수정되어 해결됨

3. **메모리 부족 오류**
   **해결:** 테스트 모드 활성화 또는 참가자 수 줄이기

4. **의존성 라이브러리 오류**
   **해결:** 필요한 라이브러리 설치
   ```bash
   pip install mne numpy scipy pywt matplotlib
   ```

## 의존성

### 필수 라이브러리
- `mne`: EEG 데이터 처리
- `numpy`: 수치 계산
- `scipy`: 과학 계산
- `pandas`: 데이터 처리

### 선택적 라이브러리
- `pywt`: LMD-DWT 처리 (PyWavelets)
- `matplotlib`: 시각화 (선택사항)
- `sklearn`: ICA-DWT Hybrid 처리 (FastICA)

## 다음 단계

전처리가 완료되면 `../analysis_pipeline/`에서 에포크 분석을 진행할 수 있습니다.

---

**참고:** 이 파이프라인은 감정 EEG 연구를 위해 특별히 설계되었으며, 다른 EEG 연구에도 적용 가능하도록 모듈화되어 있습니다.

## 모듈 설명

### 1. `eeg_preprocessing.py` - 메인 전처리 스크립트
**전체 전처리 프로세스를 관리하는 메인 실행 파일**

#### 주요 클래스:
- **`EEGPreprocessor`**: 전체 전처리 프로세스 관리

#### 주요 기능:
- 설정 기반 초기화
- 아티팩트 제거 방법 선택
- 개별 참가자 처리
- 전체 파이프라인 실행
- 결과 저장 및 요약

#### 의존성:
- 모든 다른 모듈에 의존 (중앙 제어 역할)

### 2. `config.py` - 설정 관리 모듈
**중앙화된 설정 관리 시스템**

#### 주요 클래스:
- **`PreprocessingConfig`**: 전처리 설정 관리

#### 관리하는 설정:
- 참가자 ID 리스트
- 데이터 및 출력 디렉토리
- MUSE 채널 설정
- 필터링 파라미터
- ICA 설정
- 에포크 설정
- 마커 매핑
- 아티팩트 제거 방법

#### 의존성:
- 독립적 모듈 (다른 모든 모듈에서 import됨)

### 3. `data_loader.py` - 데이터 로딩 모듈
**데이터 파일 로딩 및 검증**

#### 주요 기능:
- CSV 파일 로딩
- 참가자별 파일 찾기
- 데이터 파일 검증
- 모든 참가자 파일 목록 생성

#### 의존성:
- 독립적 모듈 (eeg_preprocessing.py에서 import됨)

### 4. `mne_processor.py` - MNE 처리 모듈
**MNE 객체 생성 및 기본 전처리**

#### 주요 기능:
- MNE Raw 객체 생성
- 채널 정보 설정
- 마커 기반 어노테이션 생성
- 기본 필터링 적용

#### 의존성:
- marker_parser.py에 의존

### 5. `ica_processor.py` - ICA 아티팩트 제거
**Independent Component Analysis 기반 아티팩트 제거**

#### 주요 기능:
- ICA 성분 분석
- 아티팩트 성분 식별
- 시각화 및 결과 저장

#### 의존성:
- 독립적 모듈

### 6. `lmd_dwt_processor.py` - LMD-DWT 아티팩트 제거
**Local Mean Decomposition + DWT 기반 아티팩트 제거**

#### 주요 기능:
- LMD 분해
- DWT 기반 노이즈 제거
- 분산 엔트로피 기반 감지

#### 의존성:
- 독립적 모듈

### 7. `ica_dwt_hybrid_processor.py` - ICA-DWT 하이브리드
**ICA와 DWT를 결합한 하이브리드 아티팩트 제거**

#### 주요 기능:
- ICA와 DWT 순차 적용
- 자동화된 처리
- 높은 정확도 아티팩트 제거

#### 의존성:
- 독립적 모듈

### 8. `epoch_processor.py` - 에포크 생성 모듈
**에포크 생성 및 저장**

#### 주요 기능:
- 이벤트 기반 에포크 생성
- 에포크 데이터 저장
- 메타데이터 관리

#### 의존성:
- 독립적 모듈

### 9. `marker_parser.py` - 마커 파싱 모듈
**마커 데이터 파싱 및 이벤트 처리**

#### 주요 기능:
- 마커 데이터 파싱
- 이벤트 매핑
- 어노테이션 생성

#### 의존성:
- 독립적 모듈

### 10. `analysis_utils.py` - 분석 유틸리티
**분석 및 검증 유틸리티 함수들**

#### 주요 기능:
- 이벤트 카운트 분석
- 처리 결과 요약
- 파이프라인 검증

#### 의존성:
- data_loader.py, marker_parser.py에 의존