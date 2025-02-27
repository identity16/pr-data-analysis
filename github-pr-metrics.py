#!/usr/bin/env python3
import requests
import pandas as pd
import datetime
import argparse
import time
import os
import numpy as np
from dateutil.parser import parse

# GitHub API 상수
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # 환경 변수에서 토큰 가져오기

def get_pull_requests(owner, repo, state="all", per_page=100, max_pages=None):
    """
    GitHub 저장소에서 Pull Request 데이터를 가져옵니다.
    
    매개변수:
    - owner: 저장소 소유자 (사용자명 또는 조직명)
    - repo: 저장소 이름
    - state: PR 상태 ('open', 'closed', 또는 'all')
    - per_page: 페이지당 PR 수
    - max_pages: 가져올 최대 페이지 수 (None은 모든 페이지)
    
    반환값:
    - Pull Request 데이터 목록
    """
    pull_requests = []
    page = 1
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    while True:
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "direction": "desc"
        }
        
        url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls"
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"PR 가져오기 오류: {response.status_code} - {response.text}")
            break
        
        page_data = response.json()
        
        if not page_data:
            break
            
        pull_requests.extend(page_data)
        
        page += 1
        
        if max_pages and page > max_pages:
            break
            
        # GitHub API 속도 제한 준수
        time.sleep(0.5)
    
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
            "outcome": outcome
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
    
    args = parser.parse_args()
    
    # max_prs 기반으로 max_pages 계산
    max_pages = None
    if args.max_prs:
        max_pages = (args.max_prs + 99) // 100  # 100으로 나눈 올림 나눗셈
    
    # Pull Request 가져오기
    print(f"{args.owner}/{args.repo}에서 Pull Request 가져오는 중...")
    pull_requests = get_pull_requests(args.owner, args.repo, args.state, max_pages=max_pages)
    
    if args.max_prs and len(pull_requests) > args.max_prs:
        pull_requests = pull_requests[:args.max_prs]
    
    print(f"{len(pull_requests)}개의 Pull Request를 가져왔습니다.")
    
    if not pull_requests:
        print("Pull Request를 찾을 수 없습니다. 종료합니다.")
        return
    
    # 메트릭 계산
    metrics_df = calculate_pr_metrics(args.owner, args.repo, pull_requests)
    
    # 파일로 저장
    metrics_df.to_csv(args.output, index=False)
    print(f"메트릭이 {args.output}에 저장되었습니다.")
    
    # 요약 통계 출력
    print("\n요약 통계:")
    print(f"총 PR 수: {len(metrics_df)}")
    print(f"머지된 PR: {metrics_df['is_merged'].sum()} ({metrics_df['is_merged'].mean() * 100:.1f}%)")
    print(f"평균 PR 크기: {metrics_df['pr_size'].mean():.1f} 라인")
    print(f"평균 PR 기간: {metrics_df['pr_duration_hours'].mean():.1f} 시간")
    
    if not metrics_df['time_to_first_review_hours'].isna().all():
        print(f"첫 리뷰까지의 평균 시간: {metrics_df['time_to_first_review_hours'].mean():.1f} 시간")
    
    print(f"PR당 평균 리뷰 수: {metrics_df['review_count'].mean():.1f}")
    print(f"PR당 평균 코멘트 수: {metrics_df['comment_count'].mean():.1f}")
    print(f"PR당 평균 승인 리뷰어 수: {metrics_df['approved_reviewers'].apply(len).mean():.1f}")

if __name__ == "__main__":
    main()
