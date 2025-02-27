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