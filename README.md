# GitHub PR 코드리뷰 생산성 측정 및 시각화 프로젝트

이 프로젝트는 GitHub Pull Request에서 코드리뷰 생산성을 측정하고 시각화하는 도구입니다.

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. PR 지표 수집

```bash
python github-pr-metrics.py [소유자] [저장소] [옵션]
```

#### 인수 설명:
- `소유자`: 저장소 소유자(사용자 또는 조직)
- `저장소`: 저장소 이름
- `--state`: PR 상태 필터(open/closed/all, 기본값: all)
- `--max-prs`: 처리할 최대 PR 수
- `--output`: 출력 파일 경로(기본값: pr_metrics.csv)

#### 예시:
```bash
# 모든 PR 처리, 기본 출력 파일
python github-pr-metrics.py microsoft vscode

# 최근 100개의 PR만 처리하고 사용자 지정 출력 파일 사용
python github-pr-metrics.py facebook react --max-prs 100 --output react_metrics.csv
```

### 2. 시각화

```bash
python github-pr-visualize.py [입력_파일] [옵션]
```

#### 인수 설명:
- `입력_파일`: PR 지표가 포함된 CSV 파일(github-pr-metrics.py 출력)
- `--output-dir`: 차트 출력 디렉토리(기본값: charts)
- `--show`: 파일 저장 대신 차트를 화면에 표시

#### 예시:
```bash
# 기본 설정으로 차트 생성
python github-pr-visualize.py pr_metrics.csv

# 사용자 지정 출력 디렉토리 사용
python github-pr-visualize.py react_metrics.csv --output-dir react_charts
```

## GitHub API 토큰 설정

GitHub API 사용 시 속도 제한을 늘리려면 환경 변수에 GitHub 토큰을 설정하세요:

```bash
export GITHUB_TOKEN=your_github_token  # Linux/Mac
# 또는
set GITHUB_TOKEN=your_github_token  # Windows
```

## 자세한 정보

더 자세한 정보는 `github-pr-guide.md` 파일을 참조하세요. 

## 시각화 차트 종류

이 도구는 다음과 같은 다양한 시각화 차트를 제공합니다:

### PR 복잡도 지표 (PR Complexity Metrics)

PR의 복잡도를 다양한 요소를 종합적으로 고려하여 시각화합니다:

- **복잡도 점수 분포**: PR 복잡도 점수의 전체 분포를 히스토그램으로 표시
- **복잡도 범주별 PR 수**: 복잡도 범주(매우 낮음, 낮음, 중간, 높음, 매우 높음)별 PR 수 비교
- **복잡도와 처리 시간의 관계**: PR 복잡도와 처리 시간 간의 상관관계 분석
- **복잡도 구성 요소 기여도**: 복잡도가 높은 PR의 구성 요소 기여도 분석
- **복잡도 요소 간 상관관계**: 각 복잡도 요소 간의 상관관계를 히트맵으로 시각화
- **복잡도 요소별 가중치**: 복잡도 계산에 사용된 각 요소의 가중치 표시

복잡도 지표는 다음 요소를 가중치를 적용하여 계산합니다:
- **기본 지표**:
  - **변경된 파일 수** (25%): PR에서 변경된 파일의 수 (값이 클수록 많은 파일이 변경됨)
  - **코드 라인 수** (25%): 추가 및 삭제된 코드 라인의 총합 (값이 클수록 많은 코드가 변경됨)
  - **리뷰 반복 횟수** (15%): PR이 리뷰-수정 과정을 반복한 횟수 (값이 클수록 여러 번 수정됨)
  - **코멘트 수** (15%): PR에 달린 리뷰 코멘트의 수 (값이 클수록 많은 피드백이 있음)
- **고급 지표**:
  - **변경 분산도** (10%): 변경이 여러 파일에 고르게 분산된 정도 (값이 높을수록 변경이 여러 파일에 분산됨)
  - **리뷰 깊이** (5%): 리뷰당 코멘트 수의 비율로 측정한 리뷰 논의의 깊이 (값이 높을수록 깊은 논의가 필요했음)
  - **크기-반복 복잡성** (5%): PR 크기와 리뷰 반복 횟수의 조합 (큰 PR이 여러 번 반복되면 더 복잡함)

이 고급 복잡도 지표를 통해 PR의 실질적인 복잡성을 더 정확히 파악하여 리뷰 시간 예측 및 리소스 할당에 활용할 수 있습니다. 각 지표는 0-100 범위로 정규화되어 직관적인 비교가 가능합니다.

### 기타 차트

- PR 수명 주기 단계별 소요 시간 분석
- PR 처리 시간 분포
- PR 크기와 처리 시간의 관계
- 리뷰 시간 추이
- 리뷰어별 리뷰 부하
- 승인 리뷰어 분석
- 리뷰어별 승인 비율
- PR 생성 추이
- PR 크기별 결과
- PR 크기별 리뷰 반복 횟수
- PR당 리뷰어 수
- PR 처리량 추이
- PR 크기별 리뷰 코멘트 수
- PR 크기 분포 