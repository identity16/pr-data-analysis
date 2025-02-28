#!/usr/bin/env python3
import requests
import pandas as pd
import datetime
import argparse
import time
import os
import numpy as np
from dateutil.parser import parse
from dotenv import load_dotenv
# 코드 리뷰 분류기 모듈 임포트
import code_review_classifier

# .env 파일 로드
load_dotenv()

# GitHub API 상수
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # 환경 변수에서 토큰 가져오기
API_WAIT_TIME = float(os.environ.get("API_WAIT_TIME", "0.5"))  # API 요청 간 대기 시간

def get_pull_requests(owner, repo, state="all", per_page=100, max_pages=None, since=None, until=None, created_since=None):
    """
    GitHub 저장소에서 Pull Request 데이터를 가져옵니다.
    
    매개변수:
    - owner: 저장소 소유자 (사용자명 또는 조직명)
    - repo: 저장소 이름
    - state: PR 상태 ('open', 'closed', 또는 'all')
    - per_page: 페이지당 PR 수
    - max_pages: 가져올 최대 페이지 수 (None은 모든 페이지)
    - since: 이 날짜 이후에 업데이트된 PR만 가져옵니다 (ISO 8601 형식: YYYY-MM-DD)
      참고: GitHub API의 since는 PR 생성 날짜가 아닌 업데이트 날짜를 기준으로 합니다.
    - until: 이 날짜 이전에 생성된 PR만 가져옵니다 (ISO 8601 형식: YYYY-MM-DD)
    - created_since: 이 날짜 이후에 생성된 PR만 가져옵니다 (ISO 8601 형식: YYYY-MM-DD)
      참고: 이 필터링은 API 호출 후 클라이언트 측에서 수행됩니다.
    
    반환값:
    - Pull Request 데이터 목록
    """
    print(f"\n[로그] PR 가져오기 시작: {owner}/{repo}")
    print(f"[로그] 매개변수: state={state}, per_page={per_page}, max_pages={max_pages}")
    print(f"[로그] 날짜 필터: since={since}, until={until}, created_since={created_since}")
    
    pull_requests = []
    page = 1
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    if GITHUB_TOKEN:
        print("[로그] GitHub 토큰이 설정되었습니다.")
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    else:
        print("[로그] 경고: GitHub 토큰이 설정되지 않았습니다. API 속도 제한이 더 엄격합니다.")
    
    # created_since 매개변수가 제공된 경우 datetime 객체로 변환
    created_since_date = None
    if created_since:
        try:
            # ISO 8601 형식인 경우
            if 'T' in created_since:
                if 'Z' in created_since:
                    created_since_date = datetime.datetime.fromisoformat(created_since.replace('Z', '+00:00'))
                    print(f"[로그] created_since 변환 (ISO 8601 + Z): {created_since_date}")
                elif '+' in created_since or '-' in created_since[10:]:  # 타임존 정보가 있는지 확인
                    created_since_date = datetime.datetime.fromisoformat(created_since)
                    print(f"[로그] created_since 변환 (ISO 8601 + 타임존): {created_since_date}")
                else:
                    # 시간은 있지만 타임존 정보가 없는 경우
                    created_since_date = datetime.datetime.fromisoformat(created_since)
                    created_since_date = created_since_date.replace(tzinfo=datetime.timezone.utc)
                    print(f"[로그] created_since 변환 (ISO 8601 타임존 없음 -> UTC 추가): {created_since_date}")
            else:
                # YYYY-MM-DD 형식인 경우
                created_since_date = datetime.datetime.strptime(created_since, "%Y-%m-%d")
                created_since_date = created_since_date.replace(tzinfo=datetime.timezone.utc)
                print(f"[로그] created_since 변환 (YYYY-MM-DD -> ISO 8601 + UTC): {created_since_date}")
        except ValueError as e:
            print(f"[로그] 오류: created_since 날짜 형식 오류: {e}")
            print("날짜 형식 오류 (created_since): {e}")
            print("YYYY-MM-DD 또는 ISO 8601 형식을 사용하세요.")
            return []
    
    # 페이지 내 모든 PR이 created_since보다 오래된 경우 페이징 중단 플래그
    stop_paging = False
    
    while True:
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "direction": "desc"
        }
        
        # 날짜 필터링 매개변수 추가
        if since:
            # ISO 8601 형식으로 변환
            try:
                # ISO 8601 형식인 경우
                if 'T' in since:
                    if 'Z' in since:
                        # 이미 ISO 8601 형식이므로 그대로 사용
                        params["since"] = since
                        print(f"[로그] since 매개변수 사용 (원본 ISO 8601 + Z): {since}")
                    elif '+' in since or '-' in since[10:]:  # 타임존 정보가 있는지 확인
                        # 이미 타임존 정보가 있는 ISO 8601 형식
                        params["since"] = since
                        print(f"[로그] since 매개변수 사용 (원본 ISO 8601 + 타임존): {since}")
                    else:
                        # 시간은 있지만 타임존 정보가 없는 경우
                        since_date = datetime.datetime.fromisoformat(since)
                        since_date = since_date.replace(tzinfo=datetime.timezone.utc)
                        params["since"] = since_date.isoformat()
                        print(f"[로그] since 매개변수 변환 (타임존 추가): {params['since']}")
                else:
                    # YYYY-MM-DD 형식을 ISO 8601 형식으로 변환
                    since_date = datetime.datetime.strptime(since, "%Y-%m-%d")
                    since_date = since_date.replace(tzinfo=datetime.timezone.utc)
                    params["since"] = since_date.isoformat()
                    print(f"[로그] since 매개변수 변환 (YYYY-MM-DD -> ISO 8601): {params['since']}")
            except ValueError as e:
                print(f"[로그] 오류: since 날짜 형식 오류: {e}")
                print(f"날짜 형식 오류 (since): {e}")
                print("YYYY-MM-DD 또는 ISO 8601 형식을 사용하세요.")
                return []
        
        url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls"
        print(f"[로그] API 요청: {url} (페이지 {page})")
        print(f"[로그] 요청 매개변수: {params}")
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"[로그] API 오류: {response.status_code} - {response.text}")
            print(f"PR 가져오기 오류: {response.status_code} - {response.text}")
            break
        
        page_data = response.json()
        print(f"[로그] 페이지 {page}에서 {len(page_data)}개의 PR을 가져왔습니다.")
        
        if not page_data:
            print(f"[로그] 더 이상 PR이 없습니다. 페이지 {page}에서 종료합니다.")
            break
        
        # until 매개변수가 제공된 경우 수동으로 필터링
        if until or created_since_date:
            original_count = len(page_data)
            # until 매개변수 처리
            if until:
                try:
                    # ISO 8601 형식인 경우
                    if 'T' in until:
                        if 'Z' in until:
                            until_date = datetime.datetime.fromisoformat(until.replace('Z', '+00:00'))
                            print(f"[로그] until 변환 (ISO 8601 + Z): {until_date}")
                        elif '+' in until or '-' in until[10:]:  # 타임존 정보가 있는지 확인
                            until_date = datetime.datetime.fromisoformat(until)
                            print(f"[로그] until 변환 (ISO 8601 + 타임존): {until_date}")
                        else:
                            # 시간은 있지만 타임존 정보가 없는 경우
                            until_date = datetime.datetime.fromisoformat(until)
                            until_date = until_date.replace(tzinfo=datetime.timezone.utc)
                            print(f"[로그] until 변환 (ISO 8601 타임존 없음 -> UTC 추가): {until_date}")
                    else:
                        # YYYY-MM-DD 형식인 경우
                        until_date = datetime.datetime.strptime(until, "%Y-%m-%d")
                        # 종료일은 해당 일의 끝(23:59:59)까지 포함
                        until_date = until_date.replace(hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc)
                        print(f"[로그] until 변환 (YYYY-MM-DD -> ISO 8601 + UTC, 23:59:59): {until_date}")
                except ValueError as e:
                    print(f"[로그] 오류: until 날짜 형식 오류: {e}")
                    print(f"날짜 형식 오류 (until): {e}")
                    print("YYYY-MM-DD 또는 ISO 8601 형식을 사용하세요.")
                    return []
            else:
                until_date = None
                print("[로그] until 필터가 제공되지 않았습니다.")
            
            filtered_data = []
            filtered_out_count = 0
            
            # 페이지 내 모든 PR이 created_since보다 오래된지 확인하기 위한 카운터
            all_prs_too_old = True if created_since_date else False
            
            for pr in page_data:
                # PR의 생성 날짜에서 타임존 정보 추출
                created_at_str = pr['created_at']
                try:
                    # GitHub API는 ISO 8601 형식으로 날짜를 반환합니다 (예: 2023-01-01T12:00:00Z)
                    # Z는 UTC 타임존을 의미합니다
                    if 'Z' in created_at_str:
                        # Z를 +00:00으로 변환하여 타임존 정보를 명시적으로 포함
                        created_at = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    else:
                        # 이미 타임존 정보가 있는 경우
                        created_at = datetime.datetime.fromisoformat(created_at_str)
                except ValueError:
                    # 다른 형식인 경우 직접 파싱
                    created_at = datetime.datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
                    # 타임존 정보 추가
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                
                # until 필터링
                if until_date and created_at > until_date:
                    filtered_out_count += 1
                    continue
                
                # created_since 필터링
                if created_since_date:
                    if created_at < created_since_date:
                        filtered_out_count += 1
                        continue
                    else:
                        # 하나라도 created_since_date 이후의 PR이 있으면 all_prs_too_old는 False
                        all_prs_too_old = False
                
                filtered_data.append(pr)
            
            page_data = filtered_data
            print(f"[로그] 클라이언트 측 필터링 결과: {original_count}개 중 {len(page_data)}개 유지, {filtered_out_count}개 필터링됨")
            
            # 페이지 내 모든 PR이 created_since보다 오래된 경우 페이징 중단
            if created_since_date and all_prs_too_old:
                print(f"[로그] 페이지 {page}의 모든 PR이 {created_since_date}보다 오래되었습니다. 페이징을 중단합니다.")
                stop_paging = True
        
        pull_requests.extend(page_data)
        
        page += 1
        
        # 페이징 중단 조건 확인
        if stop_paging:
            print("[로그] 페이징 중단: 더 이상의 PR은 날짜 필터 조건을 만족하지 않습니다.")
            break
        
        if max_pages and page > max_pages:
            print(f"[로그] 최대 페이지 수({max_pages})에 도달했습니다. 페이징을 중단합니다.")
            break
            
        # GitHub API 속도 제한 준수
        print(f"[로그] API 속도 제한 준수를 위해 {API_WAIT_TIME}초 대기")
        time.sleep(API_WAIT_TIME)
    
    print(f"[로그] 총 {len(pull_requests)}개의 PR을 가져왔습니다.")
    return pull_requests

def get_pr_reviews(owner, repo, pr_number):
    """
    특정 Pull Request의 리뷰 데이터를 가져옵니다.
    
    매개변수:
    - owner: 저장소 소유자
    - repo: 저장소 이름
    - pr_number: Pull Request 번호
    
    반환값:
    - 리뷰 데이터 목록
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"PR #{pr_number}의 리뷰 가져오기 오류: {response.status_code} - {response.text}")
        return []
    
    # GitHub API 속도 제한 준수
    time.sleep(0.5)
    
    return response.json()

def get_pr_comments(owner, repo, pr_number):
    """
    특정 Pull Request의 코멘트 데이터를 가져옵니다.
    
    매개변수:
    - owner: 저장소 소유자
    - repo: 저장소 이름
    - pr_number: Pull Request 번호
    
    반환값:
    - 코멘트 데이터 목록
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"PR #{pr_number}의 코멘트 가져오기 오류: {response.status_code} - {response.text}")
        return []
    
    # GitHub API 속도 제한 준수
    time.sleep(0.5)
    
    return response.json()

def get_pr_commits(owner, repo, pr_number):
    """
    특정 Pull Request의 커밋 데이터를 가져옵니다.
    
    매개변수:
    - owner: 저장소 소유자
    - repo: 저장소 이름
    - pr_number: Pull Request 번호
    
    반환값:
    - 커밋 데이터 목록
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pr_number}/commits"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"PR #{pr_number}의 커밋 가져오기 오류: {response.status_code} - {response.text}")
        return []
    
    # GitHub API 속도 제한 준수
    time.sleep(0.5)
    
    return response.json()

def get_pr_details(owner, repo, pr_number):
    """
    특정 Pull Request의 세부 정보를 가져옵니다.
    
    매개변수:
    - owner: 저장소 소유자
    - repo: 저장소 이름
    - pr_number: Pull Request 번호
    
    반환값:
    - PR 세부 정보
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"PR #{pr_number}의 세부 정보 가져오기 오류: {response.status_code} - {response.text}")
        return {}
    
    # GitHub API 속도 제한 준수
    time.sleep(0.5)
    
    return response.json()

def calculate_pr_metrics(owner, repo, pull_requests):
    """
    각 Pull Request의 메트릭을 계산합니다.
    
    매개변수:
    - owner: 저장소 소유자
    - repo: 저장소 이름
    - pull_requests: Pull Request 데이터 목록
    
    반환값:
    - PR 메트릭이 포함된 DataFrame
    """
    metrics_data = []
    
    total_prs = len(pull_requests)
    print(f"{total_prs}개의 Pull Request 처리 중...")
    
    for i, pr in enumerate(pull_requests):
        print(f"PR #{pr['number']} 처리 중 ({i+1}/{total_prs})...")
        
        # PR 세부 정보 가져오기
        pr_details = get_pr_details(owner, repo, pr['number'])
        if pr_details:
            # PR 세부 정보가 있으면 업데이트
            pr.update(pr_details)
        
        # 기본 PR 데이터
        pr_number = pr['number']
        pr_title = pr['title']
        pr_author = pr['user']['login']
        pr_state = pr['state']
        
        created_at = parse(pr['created_at'])
        updated_at = parse(pr['updated_at'])
        
        # 머지된 PR 처리
        if pr['merged_at']:
            merged_at = parse(pr['merged_at'])
            merged_by = pr.get('merged_by', {}).get('login') if pr.get('merged_by') else None
            is_merged = True
        else:
            merged_at = None
            merged_by = None
            is_merged = False
        
        # 닫힌 PR 처리
        if pr['closed_at']:
            closed_at = parse(pr['closed_at'])
            is_closed = True
        else:
            closed_at = None
            is_closed = False
        
        # PR 기간 계산
        if merged_at:
            pr_duration = (merged_at - created_at).total_seconds() / 3600  # 시간 단위
        elif closed_at:
            pr_duration = (closed_at - created_at).total_seconds() / 3600  # 시간 단위
        else:
            pr_duration = (datetime.datetime.now(datetime.timezone.utc) - created_at).total_seconds() / 3600  # 시간 단위
        
        # 추가, 삭제, 변경된 파일 가져오기
        additions = pr.get('additions', 0)
        deletions = pr.get('deletions', 0)
        changed_files = pr.get('changed_files', 0)
        
        # PR 크기(변경된 총 라인 수) 계산
        pr_size = additions + deletions
        
        # PR 리뷰 가져오기
        reviews = get_pr_reviews(owner, repo, pr_number)
        
        # 리뷰 메트릭 계산
        review_count = len(reviews)
        reviewers = set(review['user']['login'] for review in reviews if review['user']['login'] != pr_author)
        reviewer_count = len(reviewers)
        
        # 승인/거부 계산
        approvals = sum(1 for review in reviews if review['state'] == 'APPROVED')
        rejections = sum(1 for review in reviews if review['state'] == 'CHANGES_REQUESTED')
        comments_reviews = sum(1 for review in reviews if review['state'] == 'COMMENTED')
        
        # 승인한 리뷰어 목록 (중복 제거)
        approved_reviewers = set(review['user']['login'] for review in reviews 
                               if review['state'] == 'APPROVED' and review['user']['login'] != pr_author)
        
        # 변경 요청한 리뷰어 목록 (중복 제거)
        requested_changes_reviewers = set(review['user']['login'] for review in reviews 
                                       if review['state'] == 'CHANGES_REQUESTED' and review['user']['login'] != pr_author)
        
        # 첫 리뷰까지의 시간 계산
        if reviews:
            first_review_time = min(parse(review['submitted_at']) for review in reviews)
            time_to_first_review = (first_review_time - created_at).total_seconds() / 3600  # 시간 단위
        else:
            time_to_first_review = None
        
        # PR 코멘트 가져오기
        comments = get_pr_comments(owner, repo, pr_number)
        comment_count = len(comments)
        
        # 코멘트 작성자 (PR 작성자 제외)
        comment_authors = set(comment['user']['login'] for comment in comments if comment['user']['login'] != pr_author)
        comment_author_count = len(comment_authors)
        
        # PR 커밋 가져오기
        commits = get_pr_commits(owner, repo, pr_number)
        commit_count = len(commits)
        
        # 리뷰 반복 횟수 계산 (PR 생성 이후 커밋에서 근사치)
        if commits:
            # 날짜별로 커밋 정렬
            commit_dates = [parse(commit['commit']['committer']['date']) for commit in commits]
            commits_after_creation = sum(1 for date in commit_dates if date > created_at)
            review_iterations = max(1, commits_after_creation)
        else:
            review_iterations = 1
        
        # 최종 결과 계산
        if is_merged:
            outcome = "Merged"
        elif is_closed:
            outcome = "Closed without merge"
        else:
            outcome = "Open"
        
        # 코드 리뷰 분류 수행
        print(f"PR #{pr_number}의 코드 리뷰 분류 중...")
        review_classification_metrics = code_review_classifier.get_review_classification_metrics(reviews, comments)
        
        # 메트릭 딕셔너리 생성
        metrics = {
            "pr_number": pr_number,
            "pr_title": pr_title,
            "pr_author": pr_author,
            "pr_state": pr_state,
            "created_at": created_at,
            "updated_at": updated_at,
            "merged_at": merged_at,
            "closed_at": closed_at,
            "merged_by": merged_by,
            "is_merged": is_merged,
            "is_closed": is_closed,
            "pr_duration_hours": pr_duration,
            "additions": additions,
            "deletions": deletions,
            "changed_files": changed_files,
            "pr_size": pr_size,
            "review_count": review_count,
            "reviewer_count": reviewer_count,
            "reviewers": list(reviewers),
            "approved_reviewers": list(approved_reviewers),
            "requested_changes_reviewers": list(requested_changes_reviewers),
            "approvals": approvals,
            "rejections": rejections,
            "comments_reviews": comments_reviews,
            "time_to_first_review_hours": time_to_first_review,
            "comment_count": comment_count,
            "comment_authors": list(comment_authors),
            "comment_author_count": comment_author_count,
            "commit_count": commit_count,
            "review_iterations": review_iterations,
            "outcome": outcome,
            # 코드 리뷰 분류 메트릭 추가
            **review_classification_metrics
        }
        
        metrics_data.append(metrics)
    
    # DataFrame으로 변환
    df = pd.DataFrame(metrics_data)
    
    # 계산된 필드 추가
    # 날짜만 쉽게 집계할 수 있도록 변환
    for date_col in ['created_at', 'updated_at', 'merged_at', 'closed_at']:
        if date_col in df.columns and not df[date_col].isna().all():
            df[f"{date_col}_date"] = df[date_col].dt.date
    
    # 시간 기반 분석을 위한 주 및 월 추가
    if 'created_at' in df.columns:
        df['created_week'] = df['created_at'].dt.isocalendar().week
        df['created_month'] = df['created_at'].dt.month
        df['created_year'] = df['created_at'].dt.year
        df['created_yearmonth'] = df['created_at'].dt.strftime('%Y-%m')
    
    return df

def main():
    parser = argparse.ArgumentParser(description="GitHub Pull Request에서 코드 리뷰 메트릭 추출")
    parser.add_argument("owner", help="저장소 소유자 (사용자명 또는 조직명)")
    parser.add_argument("repo", help="저장소 이름")
    parser.add_argument("--state", default="all", choices=["open", "closed", "all"],
                        help="상태별 PR 필터링 (기본값: all)")
    parser.add_argument("--max-prs", type=int, help="처리할 최대 PR 수")
    parser.add_argument("--output", default="pr_metrics.csv", help="출력 파일 경로 (기본값: pr_metrics.csv)")
    parser.add_argument("--start-date", help="시작 날짜 (YYYY-MM-DD 형식) - PR 생성 날짜 기준")
    parser.add_argument("--end-date", help="종료 날짜 (YYYY-MM-DD 형식) - PR 생성 날짜 기준")
    parser.add_argument("--updated-since", help="이 날짜 이후에 업데이트된 PR만 가져옵니다 (YYYY-MM-DD 형식)")
    parser.add_argument("--classify-reviews", action="store_true", help="코드 리뷰 분류 활성화 (OpenAI API 키 필요)")
    
    args = parser.parse_args()
    
    # OpenAI API 키 확인
    if args.classify_reviews and not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("코드 리뷰 분류를 사용하려면 OpenAI API 키를 설정하세요.")
        print("예: export OPENAI_API_KEY='your-api-key'")
    
    # max_prs 기반으로 max_pages 계산
    max_pages = None
    if args.max_prs:
        max_pages = (args.max_prs + 99) // 100  # 100으로 나눈 올림 나눗셈
    
    # Pull Request 가져오기
    print(f"{args.owner}/{args.repo}에서 Pull Request 가져오는 중...")
    pull_requests = get_pull_requests(
        args.owner, 
        args.repo, 
        args.state, 
        max_pages=max_pages, 
        since=args.updated_since,  # 업데이트 날짜 기준 필터링
        until=args.end_date,       # 생성 날짜 기준 필터링 (종료일)
        created_since=args.start_date  # 생성 날짜 기준 필터링 (시작일)
    )
    
    if args.max_prs and len(pull_requests) > args.max_prs:
        pull_requests = pull_requests[:args.max_prs]
    
    print(f"총 {len(pull_requests)}개의 Pull Request를 가져왔습니다.")
    
    # 날짜 범위 정보 출력
    date_filters = []
    if args.start_date:
        date_filters.append(f"생성 날짜 >= {args.start_date}")
    if args.end_date:
        date_filters.append(f"생성 날짜 <= {args.end_date}")
    if args.updated_since:
        date_filters.append(f"업데이트 날짜 >= {args.updated_since}")
    
    if date_filters:
        print(f"적용된 필터: {', '.join(date_filters)}")
        print(f"필터링된 Pull Request: {len(pull_requests)}개")
    
    # PR이 없는 경우 처리
    if not pull_requests:
        print("Pull Request를 찾을 수 없습니다. 종료합니다.")
        return
    
    # 메트릭 계산
    print("PR 메트릭 계산 중...")
    df = calculate_pr_metrics(args.owner, args.repo, pull_requests)
    
    # 파일로 저장
    df.to_csv(args.output, index=False)
    print(f"메트릭이 {args.output}에 저장되었습니다.")
    
    # 요약 통계 출력
    print("\n요약 통계:")
    print(f"총 PR 수: {len(df)}")
    print(f"머지된 PR: {df['is_merged'].sum()} ({df['is_merged'].mean() * 100:.1f}%)")
    print(f"평균 PR 크기: {df['pr_size'].mean():.1f} 라인")
    print(f"평균 PR 기간: {df['pr_duration_hours'].mean():.1f} 시간")
    
    if not df['time_to_first_review_hours'].isna().all():
        print(f"첫 리뷰까지의 평균 시간: {df['time_to_first_review_hours'].mean():.1f} 시간")
    
    print(f"PR당 평균 리뷰 수: {df['review_count'].mean():.1f}")
    print(f"PR당 평균 코멘트 수: {df['comment_count'].mean():.1f}")
    print(f"PR당 평균 승인 리뷰어 수: {df['approved_reviewers'].apply(len).mean():.1f}")
    
    # 코드 리뷰 분류 통계 출력
    print("\n코드 리뷰 분류 통계:")
    review_category_columns = [col for col in df.columns if col.startswith('review_category_')]
    if review_category_columns:
        for col in review_category_columns:
            # 카테고리 이름 추출 및 포맷팅
            category_name = col.replace('review_category_', '').replace('_', ' ')
            category_name = ' '.join(word.capitalize() for word in category_name.split())
            total_count = df[col].sum()
            avg_per_pr = df[col].mean()
            print(f"{category_name}: 총 {total_count}개 (PR당 평균 {avg_per_pr:.1f}개)")
    else:
        print("코드 리뷰 분류 데이터가 없습니다.")

if __name__ == "__main__":
    main()
