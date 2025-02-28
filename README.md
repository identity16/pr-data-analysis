# GitHub PR 데이터 분석 도구

이 도구는 GitHub 저장소의 Pull Request 데이터를 수집하고 분석하여 코드 리뷰 메트릭을 추출합니다. LLM(대규모 언어 모델)을 사용하여 코드 리뷰 코멘트를 분류하는 기능도 포함되어 있습니다.

## 기능

- GitHub API를 사용하여 PR 데이터 수집
- PR 기간, 크기, 리뷰 수, 코멘트 수 등 다양한 메트릭 계산
- LLM을 사용한 코드 리뷰 코멘트 분류
- 결과를 CSV 파일로 저장
- 요약 통계 출력

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음과 같이 설정합니다:

```
# GitHub 토큰 (선택 사항이지만 API 속도 제한을 높이기 위해 권장)
GITHUB_TOKEN=your-github-token

# 코드 리뷰 분류를 위한 OpenAI API 설정 (선택 사항)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
```

또는 환경 변수를 직접 설정할 수도 있습니다:

```bash
export GITHUB_TOKEN='your-github-token'
export OPENAI_API_KEY='your-openai-api-key'
```

## 사용 방법

기본 사용법:

```bash
python github-pr-metrics.py 소유자 저장소
```

예시:

```bash
python github-pr-metrics.py microsoft vscode
```

### 옵션

- `--state`: PR 상태 필터링 (`open`, `closed`, `all`, 기본값: `all`)
- `--max-prs`: 처리할 최대 PR 수
- `--output`: 출력 파일 경로 (기본값: `pr_metrics.csv`)
- `--start-date`: 시작 날짜 (YYYY-MM-DD 형식)
- `--end-date`: 종료 날짜 (YYYY-MM-DD 형식)
- `--updated-since`: 이 날짜 이후에 업데이트된 PR만 가져옵니다 (YYYY-MM-DD 형식)
- `--classify-reviews`: 코드 리뷰 분류 활성화 (OpenAI API 키 필요)

예시:

```bash
python github-pr-metrics.py microsoft vscode --state closed --max-prs 100 --start-date 2023-01-01 --end-date 2023-12-31 --classify-reviews
```

## 코드 리뷰 분류 카테고리

코드 리뷰 분류기는 다음 카테고리를 사용합니다:

- 버그 수정 제안
- 코드 품질 개선
- 성능 최적화
- 보안 이슈
- 기능 제안
- 문서화 요청
- 테스트 관련
- 스타일 가이드 준수
- 아키텍처 개선
- 기타

## 출력 파일 형식

출력 CSV 파일에는 다음과 같은 열이 포함됩니다:

- 기본 PR 정보 (번호, 제목, 작성자, 상태 등)
- 날짜 정보 (생성, 업데이트, 머지, 닫힘)
- PR 크기 메트릭 (추가, 삭제, 변경된 파일 수)
- 리뷰 메트릭 (리뷰 수, 리뷰어 수, 승인 수 등)
- 코멘트 메트릭 (코멘트 수, 코멘트 작성자 수)
- 커밋 메트릭 (커밋 수, 리뷰 반복 횟수)
- 코드 리뷰 분류 메트릭 (카테고리별 코드 리뷰 수)

## 주의 사항

- GitHub API는 속도 제한이 있습니다. 토큰을 사용하면 제한이 높아집니다.
- 대량의 PR을 처리할 때는 `--max-prs` 옵션을 사용하여 처리할 PR 수를 제한하는 것이 좋습니다.
- 코드 리뷰 분류 기능을 사용하려면 OpenAI API 키가 필요하며, API 사용에 따른 비용이 발생할 수 있습니다. 