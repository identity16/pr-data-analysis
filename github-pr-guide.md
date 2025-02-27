# GitHub PR 코드리뷰 생산성 측정 및 시각화 가이드

이 문서는 GitHub Pull Request에서 코드리뷰 생산성을 측정하고 시각화하는 스크립트의 사용 방법과 주요 기능을 설명합니다.

## 개요

이 프로젝트는 다음 두 개의 주요 Python 스크립트로 구성되어 있습니다:

1. **`github-pr-metrics.py`**: GitHub API를 통해 Pull Request 데이터를 수집하고 코드리뷰 관련 지표를 계산합니다.
2. **`github-pr-visualize.py`**: 수집된 데이터를 기반으로 다양한 차트를 생성합니다.

## 요구사항

- Python 3.6 이상
- 필요한 패키지: `requests`, `pandas`, `matplotlib`, `seaborn`, `numpy`
- GitHub API 토큰(선택 사항이지만 API 속도 제한을 늘리기 위해 권장됨)

패키지 설치:
```bash
pip install requests pandas matplotlib seaborn numpy
```

## GitHub API 토큰 설정

GitHub API 사용 시 속도 제한을 늘리려면 환경 변수에 GitHub 토큰을 설정하세요:

```bash
export GITHUB_TOKEN=your_github_token
```

## 1. 코드리뷰 지표 추출 스크립트 (github-pr-metrics.py)

### 주요 기능

- GitHub 저장소에서 Pull Request 데이터 수집
- PR 리뷰, 코멘트, 커밋 정보 수집
- 다양한 코드리뷰 생산성 지표 계산
- 결과를 CSV 파일로 저장
- 특정 기간의 PR 데이터만 필터링하여 분석

### 추출되는 주요 지표

| 지표                          | 설명                                      |
| ----------------------------- | ----------------------------------------- |
| `pr_number`                   | Pull Request 번호                         |
| `pr_title`                    | Pull Request 제목                         |
| `pr_author`                   | Pull Request 작성자                       |
| `pr_state`                    | Pull Request 상태(open/closed)            |
| `created_at`                  | 생성 시간                                 |
| `merged_at`                   | 병합 시간                                 |
| `pr_duration_hours`           | PR 생성부터 병합/종료까지 소요 시간(시간) |
| `additions`                   | 추가된 라인 수                            |
| `deletions`                   | 삭제된 라인 수                            |
| `changed_files`               | 변경된 파일 수                            |
| `pr_size`                     | PR 크기(변경된 총 라인 수)                |
| `review_count`                | 받은 리뷰 수                              |
| `reviewer_count`              | 리뷰어 수                                 |
| `reviewers`                   | 리뷰어 목록                               |
| `approved_reviewers`          | 승인한 리뷰어 목록                        |
| `requested_changes_reviewers` | 변경 요청한 리뷰어 목록                   |
| `approvals`                   | 승인 수                                   |
| `rejections`                  | 변경 요청 수                              |
| `time_to_first_review_hours`  | 첫 리뷰까지 걸린 시간(시간)               |
| `comment_count`               | 코멘트 수                                 |
| `commit_count`                | 커밋 수                                   |
| `review_iterations`           | 리뷰 반복 횟수(근사치)                    |
| `outcome`                     | PR 결과(Merged/Closed without merge/Open) |

### 사용 방법

```bash
python github-pr-metrics.py [소유자] [저장소] [옵션]
```

#### 인수 설명:
- `소유자`: 저장소 소유자(사용자 또는 조직)
- `저장소`: 저장소 이름
- `--state`: PR 상태 필터(open/closed/all, 기본값: all)
- `--max-prs`: 처리할 최대 PR 수
- `--output`: 출력 파일 경로(기본값: pr_metrics.csv)
- `--start-date`: 시작 날짜(YYYY-MM-DD 형식)
- `--end-date`: 종료 날짜(YYYY-MM-DD 형식)

#### 예시:
```bash
# 모든 PR 처리, 기본 출력 파일
python github-pr-metrics.py microsoft vscode

# 최근 100개의 PR만 처리하고 사용자 지정 출력 파일 사용
python github-pr-metrics.py facebook react --max-prs 100 --output react_metrics.csv

# 닫힌 PR만 처리
python github-pr-metrics.py google tensorflow --state closed

# 특정 기간의 PR만 처리
python github-pr-metrics.py kubernetes kubernetes --start-date 2023-01-01 --end-date 2023-12-31 --output k8s_2023_metrics.csv
```

## 2. 시각화 스크립트 (github-pr-visualize.py)

### 주요 기능

- PR 지표 데이터 로드 및 전처리
- 다양한 차트 및 시각화 생성
- 차트를 파일로 저장하거나 화면에 표시
- 특정 기간의 PR 데이터만 필터링하여 시각화

### 생성되는 차트

| 차트 이름                         | 설명                                                                   |
| --------------------------------- | ---------------------------------------------------------------------- |
| `pr_duration_histogram.png`       | PR 생성부터 병합/종료까지 소요 시간 분포                               |
| `pr_size_vs_duration.png`         | PR 크기와 생명주기 소요 시간의 관계                                    |
| `review_time_trend.png`           | 시간 경과에 따른 첫 리뷰 시간 추세                                     |
| `review_load_by_reviewer.png`     | 리뷰어별 리뷰 부하 상세 분석 (리뷰 유형 분포, 승인 비율, 평균 PR 크기) |
| `approved_reviewers.png`          | 승인한 리뷰어별 승인 횟수                                              |
| `approval_ratio_by_reviewer.png`  | 리뷰어별 승인 비율(%)                                                  |
| `pr_creation_over_time.png`       | 시간 경과에 따른 PR 생성 추이                                          |
| `pr_outcome_by_size.png`          | 크기 카테고리별 PR 결과 비율                                           |
| `review_iterations_by_size.png`   | PR 크기별 리뷰 반복 횟수                                               |
| `reviewers_per_pr.png`            | PR당 리뷰어 수 분포                                                    |
| `pr_throughput_over_time.png`     | 시간에 따른 PR 처리량                                                  |
| `review_comments_per_pr_size.png` | PR 크기별 리뷰 코멘트 수                                               |
| `pr_size_distribution.png`        | PR 크기 분포                                                           |

### 사용 방법

```bash
python github-pr-visualize.py [입력_파일] [옵션]
```

#### 인수 설명:
- `입력_파일`: PR 지표가 포함된 CSV 파일(github-pr-metrics.py 출력)
- `--output-dir`: 차트 출력 디렉토리(기본값: charts)
- `--show`: 파일 저장 대신 차트를 화면에 표시
- `--start-date`: 시작 날짜(YYYY-MM-DD 형식)
- `--end-date`: 종료 날짜(YYYY-MM-DD 형식)

#### 예시:
```bash
# 기본 설정으로 차트 생성
python github-pr-visualize.py pr_metrics.csv

# 사용자 지정 출력 디렉토리 사용
python github-pr-visualize.py react_metrics.csv --output-dir react_charts

# 차트를 화면에 표시
python github-pr-visualize.py tensorflow_metrics.csv --show

# 특정 기간의 PR만 시각화
python github-pr-visualize.py pr_metrics.csv --start-date 2023-01-01 --end-date 2023-12-31 --output-dir charts_2023
```

## 전체 워크플로우 예시

다음은 전체 워크플로우의 예시입니다:

```bash
# 1. GitHub API 토큰 설정(선택 사항)
export GITHUB_TOKEN=your_github_token

# 2. PR 지표 수집
python github-pr-metrics.py kubernetes kubernetes --max-prs 200 --output k8s_metrics.csv

# 3. 차트 생성
python github-pr-visualize.py k8s_metrics.csv --output-dir k8s_charts

# 4. 특정 연도의 PR 지표 수집
python github-pr-metrics.py kubernetes kubernetes --start-date 2023-01-01 --end-date 2023-12-31 --output k8s_2023_metrics.csv

# 5. 2023년 데이터의 차트 생성
python github-pr-visualize.py k8s_2023_metrics.csv --output-dir k8s_2023_charts
```

## 제한 사항

- GitHub API는 인증되지 않은 요청에 대해 시간당 60회, 인증된 요청에 대해 시간당 5,000회의 요청 제한이 있습니다.
- 매우 큰 저장소의 경우 모든 PR을 처리하는 데 시간이 오래 걸릴 수 있으므로 `--max-prs` 옵션을 사용하여 최근 PR만 분석하는 것이 좋습니다.
- 이 스크립트는 PR 생산성에 대한 대략적인 측정치를 제공하며, 정확한 값은 GitHub의 데이터 가용성에 따라 달라질 수 있습니다.

## 팁과 모범 사례

### 코드 리뷰 생산성 향상을 위한 제안
1. **작은 PR 권장**: 데이터는 일반적으로 작은 PR이 더 빠르게 리뷰되고 더 높은 병합률을 갖는다는 것을 보여줍니다.
2. **리뷰 부하 균형 유지**: 특정 리뷰어에게 과도한 부담이 가지 않도록 리뷰 작업을 분산시키세요.
3. **첫 리뷰까지의 시간 개선**: 이 지표를 개선하면 전체 PR 생명주기가 단축됩니다.
4. **리뷰 일관성 유지**: 리뷰 반복 횟수와 코멘트 수를 모니터링하여 리뷰 품질과 깊이를 일관되게 유지하세요.

### 데이터 해석 시 고려사항
- 단일 지표만으로 결론을 내리지 마세요. 여러 지표를 종합적으로 고려하세요.
- 팀 구성, 프로젝트 복잡성, 조직 문화 등의 맥락을 고려하세요.
- 시간 경과에 따른 추세를 분석하여 개선 영역을 파악하세요.
