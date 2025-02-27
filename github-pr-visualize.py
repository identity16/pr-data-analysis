#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import platform
import matplotlib.font_manager as fm
from datetime import datetime

# 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    
    # 폰트 후보 목록 (시스템별로 다른 폰트 경로 사용)
    if system == 'Darwin':  # macOS
        font_candidates = [
            'AppleGothic',
            'Apple SD Gothic Neo',
            'Nanum Gothic',
            'NanumGothic',
            'NanumSquareRound',
            'Malgun Gothic',
            'Arial Unicode MS'
        ]
    elif system == 'Windows':
        font_candidates = [
            'Malgun Gothic',
            'NanumGothic',
            'Gulim',
            'Dotum',
            'Arial Unicode MS'
        ]
    else:  # Linux 등
        font_candidates = [
            'NanumGothic',
            'NanumBarunGothic',
            'UnDotum',
            'UnBatang',
            'Arial Unicode MS'
        ]
    
    # 설치된 폰트 중에서 사용 가능한 폰트 찾기
    font_found = False
    for font in font_candidates:
        try:
            fm.findfont(font, fallback_to_default=False)
            plt.rcParams['font.family'] = font
            print(f"한글 폰트 '{font}'를 사용합니다.")
            font_found = True
            break
        except:
            continue
    
    if not font_found:
        print("경고: 한글 폰트를 찾을 수 없습니다. 일부 텍스트가 깨질 수 있습니다.")
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 적용
set_korean_font()

def load_pr_data(file_path, start_date=None, end_date=None):
    """CSV 파일에서 PR 메트릭 데이터를 로드합니다."""
    df = pd.read_csv(file_path)
    
    # 날짜 문자열을 datetime 객체로 변환
    date_columns = [col for col in df.columns if col.endswith('_at')]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # 날짜 필터링
    if start_date or end_date:
        if 'created_at' in df.columns:
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['created_at'] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['created_at'] <= end_date]
    
    return df

def plot_pr_duration_histogram(df, output_file=None):
    """PR 기간 히스토그램을 그립니다."""
    plt.figure(figsize=(10, 6))
    
    # 더 나은 시각화를 위해 극단적인 이상치 필터링
    duration_data = df['pr_duration_hours']
    q1, q3 = np.percentile(duration_data, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    filtered_data = duration_data[duration_data <= upper_bound]
    
    sns.histplot(filtered_data, bins=30, kde=True)
    plt.title('PR 생성부터 병합/종료까지 소요 시간 분포')
    plt.xlabel('PR 생명주기 소요 시간 (시간)')
    plt.ylabel('건수')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_size_vs_duration(df, output_file=None):
    """PR 크기 대 기간 산점도를 그립니다."""
    plt.figure(figsize=(10, 6))
    
    # 더 나은 시각화를 위해 극단적인 이상치 필터링
    q1_size, q3_size = np.percentile(df['pr_size'], [25, 75])
    iqr_size = q3_size - q1_size
    upper_bound_size = q3_size + 3 * iqr_size
    
    q1_dur, q3_dur = np.percentile(df['pr_duration_hours'], [25, 75])
    iqr_dur = q3_dur - q1_dur
    upper_bound_dur = q3_dur + 3 * iqr_dur
    
    filtered_df = df[(df['pr_size'] <= upper_bound_size) & 
                     (df['pr_duration_hours'] <= upper_bound_dur)]
    
    sns.scatterplot(data=filtered_df, x='pr_size', y='pr_duration_hours', 
                    hue='is_merged', alpha=0.7)
    plt.title('PR 크기와 생명주기 소요 시간의 관계')
    plt.xlabel('PR 크기 (변경된 라인 수)')
    plt.ylabel('PR 생성부터 병합/종료까지 소요 시간 (시간)')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_review_time_trend(df, output_file=None):
    """시간 경과에 따른 첫 리뷰 시간 추세를 그립니다."""
    plt.figure(figsize=(12, 6))
    
    # created_at 날짜와 time_to_first_review가 있는지 확인
    if 'created_at' not in df.columns or 'time_to_first_review_hours' not in df.columns:
        print("리뷰 시간 추세에 필요한 열이 없습니다")
        return
    
    # time_to_first_review가 NaN인 행 필터링
    filtered_df = df.dropna(subset=['time_to_first_review_hours'])
    
    if filtered_df.empty:
        print("필터링 후 리뷰 시간 추세에 사용할 데이터가 없습니다")
        return
    
    # 극단적인 이상치 필터링
    q1, q3 = np.percentile(filtered_df['time_to_first_review_hours'], [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    filtered_df = filtered_df[filtered_df['time_to_first_review_hours'] <= upper_bound]
    
    # created_at으로 정렬
    filtered_df = filtered_df.sort_values('created_at')
    
    # 이동 평균 생성
    window_size = min(10, len(filtered_df))
    filtered_df['rolling_avg'] = filtered_df['time_to_first_review_hours'].rolling(window=window_size).mean()
    
    # 그래프 작성
    plt.plot(filtered_df['created_at'], filtered_df['time_to_first_review_hours'], 
             'o', alpha=0.5, label='개별 PR')
    plt.plot(filtered_df['created_at'], filtered_df['rolling_avg'], 
             'r-', linewidth=2, label=f'{window_size}-PR 이동 평균')
    
    plt.title('첫 리뷰 시간 추세')
    plt.xlabel('PR 생성 날짜')
    plt.ylabel('첫 리뷰까지의 시간 (시간)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 가독성을 위해 x축 레이블 회전
    plt.xticks(rotation=45)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_review_load_by_reviewer(df, output_file=None):
    """리뷰어별 리뷰 부하 분포를 그립니다."""
    plt.figure(figsize=(15, 10))
    
    # 리뷰어 데이터 처리
    reviewer_data = {}
    
    # 모든 리뷰 및 승인/거부 데이터 추출
    for idx, row in df.iterrows():
        # 리뷰어 리스트 처리
        reviewers_list = row.get('reviewers', [])
        if isinstance(reviewers_list, str):
            try:
                reviewers_list = eval(reviewers_list)
            except:
                reviewers_list = []
                
        # 승인 리뷰어 리스트 처리
        approved_list = row.get('approved_reviewers', [])
        if isinstance(approved_list, str):
            try:
                approved_list = eval(approved_list)
            except:
                approved_list = []
                
        # 변경 요청 리뷰어 리스트 처리
        requested_changes_list = row.get('requested_changes_reviewers', [])
        if isinstance(requested_changes_list, str):
            try:
                requested_changes_list = eval(requested_changes_list)
            except:
                requested_changes_list = []
        
        # 각 리뷰어에 대한 데이터 업데이트
        for reviewer in reviewers_list:
            if reviewer not in reviewer_data:
                reviewer_data[reviewer] = {
                    'total': 0,
                    'approved': 0,
                    'requested_changes': 0,
                    'commented': 0,
                    'pr_sizes': []
                }
            reviewer_data[reviewer]['total'] += 1
            
            # PR 크기 정보 저장
            if 'pr_size' in row:
                reviewer_data[reviewer]['pr_sizes'].append(row['pr_size'])
            
        # 승인 및 변경 요청 정보 업데이트
        for approver in approved_list:
            if approver in reviewer_data:
                reviewer_data[approver]['approved'] += 1
                
        for requester in requested_changes_list:
            if requester in reviewer_data:
                reviewer_data[requester]['requested_changes'] += 1
                
        # 코멘트만 한 경우 계산 (전체 - 승인 - 변경요청)
        for reviewer in reviewers_list:
            if reviewer in reviewer_data:
                commented = reviewer_data[reviewer]['total'] - (
                    reviewer_data[reviewer]['approved'] + reviewer_data[reviewer]['requested_changes']
                )
                reviewer_data[reviewer]['commented'] = max(0, commented)
    
    if not reviewer_data:
        print("리뷰어 데이터가 없습니다")
        return
    
    # 리뷰 수 기준으로 상위 리뷰어 선택
    sorted_reviewers = sorted(reviewer_data.items(), key=lambda x: x[1]['total'], reverse=True)
    top_n = min(15, len(sorted_reviewers))
    top_reviewers = sorted_reviewers[:top_n]
    
    # 그래프 데이터 준비
    names = [item[0] for item in top_reviewers]
    total_reviews = [item[1]['total'] for item in top_reviewers]
    approvals = [item[1]['approved'] for item in top_reviewers]
    change_requests = [item[1]['requested_changes'] for item in top_reviewers]
    comments = [item[1]['commented'] for item in top_reviewers]
    
    # 평균 PR 크기 계산
    avg_pr_sizes = []
    for item in top_reviewers:
        sizes = item[1]['pr_sizes']
        avg_size = np.mean(sizes) if sizes else 0
        avg_pr_sizes.append(avg_size)
    
    # 승인 비율 계산
    approval_ratios = []
    for item in top_reviewers:
        total = item[1]['total']
        approved = item[1]['approved']
        ratio = (approved / total * 100) if total > 0 else 0
        approval_ratios.append(ratio)
    
    # 서브플롯 생성
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))
    
    # 1. 리뷰 유형별 분포 (누적 막대 그래프)
    bar_width = 0.8
    bars1 = ax1.barh(names, approvals, bar_width, label='승인', color='green')
    bars2 = ax1.barh(names, change_requests, bar_width, left=approvals, label='변경 요청', color='red')
    bars3 = ax1.barh(names, comments, bar_width, left=np.array(approvals) + np.array(change_requests), 
                    label='코멘트만', color='gray')
    
    # 총 리뷰 수 표시
    for i, total in enumerate(total_reviews):
        ax1.text(total + 1, i, f'{total}개', va='center')
    
    ax1.set_title('리뷰어별 리뷰 유형 분포')
    ax1.set_xlabel('리뷰 수')
    ax1.set_ylabel('리뷰어')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 승인 비율
    bars = ax2.barh(names, approval_ratios)
    
    # 승인 비율 표시
    for i, ratio in enumerate(approval_ratios):
        ax2.text(ratio + 1, i, f'{ratio:.1f}%', va='center')
    
    ax2.set_title('리뷰어별 승인 비율')
    ax2.set_xlabel('승인 비율 (%)')
    ax2.set_xlim(0, 105)  # 0-100% 범위 + 텍스트 공간
    ax2.grid(True, alpha=0.3)
    
    # 3. 평균 PR 크기
    bars = ax3.barh(names, avg_pr_sizes)
    
    # 평균 PR 크기 표시
    for i, size in enumerate(avg_pr_sizes):
        ax3.text(size + 1, i, f'{size:.0f}줄', va='center')
    
    ax3.set_title('리뷰어별 평균 리뷰 PR 크기')
    ax3.set_xlabel('평균 변경 라인 수')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'리뷰어별 리뷰 부하 상세 분석 (상위 {top_n}명)', fontsize=16, y=1.02)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_approved_reviewers(df, output_file=None):
    """승인한 리뷰어별 승인 횟수를 그립니다."""
    plt.figure(figsize=(12, 8))
    
    # 모든 승인 리뷰어 추출
    all_approved_reviewers = []
    for reviewers_list in df['approved_reviewers'].dropna():
        # 문자열 형태의 리스트를 실제 리스트로 변환
        if isinstance(reviewers_list, str):
            try:
                reviewers_list = eval(reviewers_list)
            except:
                reviewers_list = []
        all_approved_reviewers.extend(reviewers_list)
    
    if not all_approved_reviewers:
        print("승인 리뷰어 데이터가 없습니다")
        return
    
    # 리뷰어별 승인 횟수 계산
    approver_counts = pd.Series(all_approved_reviewers).value_counts()
    
    # 가독성을 위해 상위 15명의 승인 리뷰어 선택 (또는 15명보다 적으면 전체)
    top_n = min(15, len(approver_counts))
    top_approvers = approver_counts.head(top_n)
    
    # 그래프 작성
    sns.barplot(y=top_approvers.index, x=top_approvers.values)
    plt.title(f'승인 리뷰어별 승인 횟수 (상위 {top_n}명)')
    plt.xlabel('승인 횟수')
    plt.ylabel('리뷰어')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def plot_approval_ratio_by_reviewer(df, output_file=None):
    """리뷰어별 승인 비율을 그립니다."""
    plt.figure(figsize=(12, 8))
    
    # 리뷰어와 승인 리뷰어 데이터 처리
    reviewer_data = {}
    
    # 모든 리뷰 및 승인 데이터 추출
    for idx, row in df.iterrows():
        # 리뷰어 리스트 처리
        reviewers_list = row.get('reviewers', [])
        if isinstance(reviewers_list, str):
            try:
                reviewers_list = eval(reviewers_list)
            except:
                reviewers_list = []
                
        # 승인 리뷰어 리스트 처리
        approved_list = row.get('approved_reviewers', [])
        if isinstance(approved_list, str):
            try:
                approved_list = eval(approved_list)
            except:
                approved_list = []
        
        # 각 리뷰어에 대한 데이터 업데이트
        for reviewer in reviewers_list:
            if reviewer not in reviewer_data:
                reviewer_data[reviewer] = {'total': 0, 'approved': 0}
            reviewer_data[reviewer]['total'] += 1
            
        for approver in approved_list:
            if approver in reviewer_data:
                reviewer_data[approver]['approved'] += 1
    
    if not reviewer_data:
        print("리뷰어 데이터가 없습니다")
        return
        
    # 승인 비율 계산
    reviewer_ratio = {
        reviewer: data['approved'] / data['total'] * 100 
        for reviewer, data in reviewer_data.items() 
        if data['total'] >= 5  # 최소 5개 이상의 리뷰를 수행한 리뷰어만 포함
    }
    
    if not reviewer_ratio:
        print("충분한 리뷰 수를 가진 리뷰어가 없습니다")
        return
    
    # 상위 15명 선정
    top_ratios = sorted(reviewer_ratio.items(), key=lambda x: x[1], reverse=True)
    top_n = min(15, len(top_ratios))
    top_reviewers = top_ratios[:top_n]
    
    # 그래프 데이터 준비
    names = [item[0] for item in top_reviewers]
    ratios = [item[1] for item in top_reviewers]
    
    # 그래프 작성
    bars = plt.barh(names, ratios)
    
    # 바에 승인 비율 표시
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{ratios[i]:.1f}%', va='center')
    
    plt.title(f'리뷰어별 승인 비율 (최소 5개 이상 리뷰, 상위 {top_n}명)')
    plt.xlabel('승인 비율 (%)')
    plt.ylabel('리뷰어')
    plt.xlim(0, 105)  # 0-100% 범위 + 텍스트 공간
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_creation_over_time(df, output_file=None):
    """시간 경과에 따른 PR 생성을 그립니다."""
    plt.figure(figsize=(14, 6))
    
    # 일별로 그룹화
    if 'created_at' in df.columns:
        df['created_date'] = df['created_at'].dt.strftime('%Y-%m-%d')
    
    if 'created_date' not in df.columns:
        print("생성 시간 열이 없습니다")
        return
    
    daily_counts = df.groupby('created_date').size()
    
    # 날짜가 너무 많으면 가독성이 떨어질 수 있으므로 데이터 양에 따라 처리
    if len(daily_counts) > 60:  # 데이터가 많으면 일부 레이블만 표시
        plt.figure(figsize=(16, 6))  # 더 넓은 그래프
        ax = daily_counts.plot(kind='bar')
        # x축 레이블 간격 조정 (모든 레이블을 표시하지 않고 일부만 표시)
        interval = max(1, len(daily_counts) // 20)  # 최대 20개 레이블 표시
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx % interval != 0:
                label.set_visible(False)
    else:
        # 데이터가 적으면 모든 레이블 표시
        daily_counts.plot(kind='bar')
    
    plt.title('일별 생성된 PR')
    plt.xlabel('날짜')
    plt.ylabel('PR 수')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()  # 레이블이 잘리지 않도록 레이아웃 조정
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_outcome_by_size(df, output_file=None):
    """크기 카테고리별 PR 결과를 그립니다."""
    plt.figure(figsize=(10, 6))
    
    # 크기 카테고리 생성
    df['size_category'] = pd.cut(
        df['pr_size'], 
        bins=[0, 10, 50, 100, 500, 1000, float('inf')],
        labels=['XS (0-10)', 'S (11-50)', 'M (51-100)', 'L (101-500)', 'XL (501-1000)', 'XXL (1000+)']
    )
    
    # 크기별 결과 수 생성
    outcome_by_size = df.groupby(['size_category', 'outcome']).size().unstack()
    
    if outcome_by_size.empty:
        print("크기별 결과에 사용할 데이터가 없습니다")
        return
    
    # NaN 값을 0으로 채우기
    outcome_by_size = outcome_by_size.fillna(0)
    
    # 백분율 계산
    outcome_pct = outcome_by_size.div(outcome_by_size.sum(axis=1), axis=0) * 100
    
    # 그래프 작성
    outcome_pct.plot(kind='bar', stacked=True)
    plt.title('크기 카테고리별 PR 결과')
    plt.xlabel('PR 크기')
    plt.ylabel('퍼센트')
    plt.legend(title='결과')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_review_iterations_by_size(df, output_file=None):
    """PR 크기별 리뷰 반복 횟수를 그립니다."""
    plt.figure(figsize=(10, 6))
    
    # 극단적인 이상치 필터링
    q1, q3 = np.percentile(df['pr_size'], [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    filtered_df = df[df['pr_size'] <= upper_bound]
    
    # 이미 생성되지 않은 경우 크기 카테고리 생성
    if 'size_category' not in filtered_df.columns:
        filtered_df['size_category'] = pd.cut(
            filtered_df['pr_size'], 
            bins=[0, 10, 50, 100, 500, 1000, float('inf')],
            labels=['XS (0-10)', 'S (11-50)', 'M (51-100)', 'L (101-500)', 'XL (501-1000)', 'XXL (1000+)']
        )
    
    # 크기별 리뷰 반복 횟수의 박스 플롯
    sns.boxplot(x='size_category', y='review_iterations', data=filtered_df)
    plt.title('PR 크기별 리뷰 반복 횟수')
    plt.xlabel('PR 크기')
    plt.ylabel('리뷰 반복 횟수')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reviewers_per_pr(df, output_file=None):
    """PR당 리뷰어 수 분포를 그립니다."""
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df['reviewer_count'], bins=range(0, max(df['reviewer_count'])+2), kde=False)
    plt.title('PR당 리뷰어 수 분포')
    plt.xlabel('리뷰어 수')
    plt.ylabel('PR 수')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, max(df['reviewer_count'])+1))
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_throughput_over_time(df, output_file=None):
    """시간에 따른 PR 처리량을 그립니다."""
    plt.figure(figsize=(12, 6))
    
    # PR이 병합되거나 닫힌 날짜 확인
    if 'merged_at' in df.columns:
        completed_df = df.dropna(subset=['merged_at']).copy()
        completed_df['completion_date'] = pd.to_datetime(completed_df['merged_at']).dt.strftime('%Y-%m')
        
        monthly_throughput = completed_df.groupby('completion_date').size()
        
        # 그래프 작성
        monthly_throughput.plot(kind='bar')
        plt.title('월별 처리된 PR (병합됨)')
        plt.xlabel('월')
        plt.ylabel('PR 수')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print("병합 날짜 정보가 없습니다")

def plot_review_comments_per_pr_size(df, output_file=None):
    """PR 크기별 리뷰 코멘트 수를 그립니다."""
    plt.figure(figsize=(10, 6))
    
    # 이미 생성되지 않은 경우 크기 카테고리 생성
    if 'size_category' not in df.columns:
        df['size_category'] = pd.cut(
            df['pr_size'], 
            bins=[0, 10, 50, 100, 500, 1000, float('inf')],
            labels=['XS (0-10)', 'S (11-50)', 'M (51-100)', 'L (101-500)', 'XL (501-1000)', 'XXL (1000+)']
        )
    
    # 크기별 코멘트 수 계산
    comments_by_size = df.groupby('size_category')['comment_count'].mean().reset_index()
    
    # 그래프 작성
    sns.barplot(x='size_category', y='comment_count', data=comments_by_size)
    plt.title('PR 크기별 평균 코멘트 수')
    plt.xlabel('PR 크기')
    plt.ylabel('평균 코멘트 수')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_size_distribution(df, output_file=None):
    """PR 크기 분포를 그립니다."""
    plt.figure(figsize=(10, 6))
    
    # 극단적인 이상치 필터링
    q1, q3 = np.percentile(df['pr_size'], [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    filtered_data = df['pr_size'][df['pr_size'] <= upper_bound]
    
    sns.histplot(filtered_data, bins=30, kde=True)
    plt.title('PR 크기 분포 (변경된 라인 수)')
    plt.xlabel('PR 크기 (라인)')
    plt.ylabel('PR 수')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='GitHub PR 데이터 시각화')
    parser.add_argument('input_file', help='PR 지표가 포함된 CSV 파일')
    parser.add_argument('--output-dir', default='charts', help='차트 출력 디렉토리')
    parser.add_argument('--show', action='store_true', help='파일 저장 대신 차트를 화면에 표시')
    parser.add_argument('--start-date', help='시작 날짜 (YYYY-MM-DD 형식)')
    parser.add_argument('--end-date', help='종료 날짜 (YYYY-MM-DD 형식)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # PR 데이터 로드 및 필터링
    df = load_pr_data(args.input_file, args.start_date, args.end_date)
    
    # 날짜 필터링이 적용된 경우 파일 이름에 날짜 접미사 추가
    date_suffix = ''
    if args.start_date:
        date_suffix += f"_from_{args.start_date}"
    if args.end_date:
        date_suffix += f"_to_{args.end_date}"
    
    # 차트 생성
    print("차트 생성 중...")
    
    plot_pr_duration_histogram(df, 
        None if args.show else os.path.join(args.output_dir, f"pr_duration_histogram{date_suffix}.png"))
    
    plot_pr_size_vs_duration(df, 
        None if args.show else os.path.join(args.output_dir, f"pr_size_vs_duration{date_suffix}.png"))
    
    plot_review_time_trend(df, 
        None if args.show else os.path.join(args.output_dir, f"review_time_trend{date_suffix}.png"))
    
    plot_review_load_by_reviewer(df, 
        None if args.show else os.path.join(args.output_dir, f"review_load_by_reviewer{date_suffix}.png"))
    
    plot_approved_reviewers(df, 
        None if args.show else os.path.join(args.output_dir, f"approved_reviewers{date_suffix}.png"))
    
    plot_approval_ratio_by_reviewer(df, 
        None if args.show else os.path.join(args.output_dir, f"approval_ratio_by_reviewer{date_suffix}.png"))
    
    plot_pr_creation_over_time(df, 
        None if args.show else os.path.join(args.output_dir, f"pr_creation_over_time{date_suffix}.png"))
    
    plot_pr_outcome_by_size(df, 
        None if args.show else os.path.join(args.output_dir, f"pr_outcome_by_size{date_suffix}.png"))
    
    plot_review_iterations_by_size(df, 
        None if args.show else os.path.join(args.output_dir, f"review_iterations_by_size{date_suffix}.png"))
    
    plot_reviewers_per_pr(df, 
        None if args.show else os.path.join(args.output_dir, f"reviewers_per_pr{date_suffix}.png"))
    
    plot_pr_throughput_over_time(df, 
        None if args.show else os.path.join(args.output_dir, f"pr_throughput_over_time{date_suffix}.png"))
    
    plot_review_comments_per_pr_size(df, 
        None if args.show else os.path.join(args.output_dir, f"review_comments_per_pr_size{date_suffix}.png"))
    
    plot_pr_size_distribution(df, 
        None if args.show else os.path.join(args.output_dir, f"pr_size_distribution{date_suffix}.png"))
    
    if not args.show:
        print(f"모든 차트가 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
