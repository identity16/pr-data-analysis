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
    plt.title('PR 생명주기 소요 시간 분포')
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
    plt.figure(figsize=(14, 6))
    
    # PR이 병합되거나 닫힌 날짜 확인
    if 'merged_at' in df.columns:
        completed_df = df.dropna(subset=['merged_at']).copy()
        completed_df['completion_date'] = pd.to_datetime(completed_df['merged_at']).dt.strftime('%Y-%m-%d')
        
        daily_throughput = completed_df.groupby('completion_date').size()
        
        # 날짜가 너무 많으면 가독성이 떨어질 수 있으므로 데이터 양에 따라 처리
        if len(daily_throughput) > 60:  # 데이터가 많으면 일부 레이블만 표시
            plt.figure(figsize=(16, 6))  # 더 넓은 그래프
            ax = daily_throughput.plot(kind='bar')
            # x축 레이블 간격 조정 (모든 레이블을 표시하지 않고 일부만 표시)
            interval = max(1, len(daily_throughput) // 20)  # 최대 20개 레이블 표시
            for idx, label in enumerate(ax.get_xticklabels()):
                if idx % interval != 0:
                    label.set_visible(False)
        else:
            # 데이터가 적으면 모든 레이블 표시
            daily_throughput.plot(kind='bar')
        
        plt.title('일별 처리된 PR (병합됨)')
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

def plot_pr_lifecycle_stages(df, output_file=None):
    """PR 수명 주기 단계별 소요 시간을 분석하여 시각화합니다."""
    plt.figure(figsize=(14, 8))
    
    # 필요한 열이 있는지 확인
    required_columns = ['created_at', 'merged_at', 'time_to_first_review_hours']
    if not all(col in df.columns for col in required_columns):
        print("PR 수명 주기 분석에 필요한 열이 없습니다")
        return
    
    # 병합된 PR만 필터링
    merged_prs = df.dropna(subset=['merged_at']).copy()
    
    if merged_prs.empty:
        print("병합된 PR 데이터가 없습니다")
        return
    
    # 첫 리뷰 시간이 없는 경우 필터링
    merged_prs = merged_prs.dropna(subset=['time_to_first_review_hours'])
    
    if merged_prs.empty:
        print("첫 리뷰 시간 데이터가 있는 병합된 PR이 없습니다")
        return
    
    # 각 단계별 시간 계산
    merged_prs['first_review_time'] = merged_prs['time_to_first_review_hours']
    
    # 전체 PR 기간
    merged_prs['total_duration'] = (merged_prs['merged_at'] - merged_prs['created_at']).dt.total_seconds() / 3600
    
    # 첫 리뷰 이후부터 병합까지의 시간
    merged_prs['review_to_merge_time'] = merged_prs['total_duration'] - merged_prs['first_review_time']
    
    # 음수 값이 있으면 0으로 설정 (데이터 오류 방지)
    merged_prs['review_to_merge_time'] = merged_prs['review_to_merge_time'].clip(lower=0)
    
    # 최근 15개 PR만 선택 (가독성을 위해)
    recent_prs = merged_prs.sort_values('merged_at', ascending=False).head(15)
    
    # PR 번호를 문자열로 변환
    recent_prs['pr_number_str'] = recent_prs['pr_number'].astype(str)
    
    # 그래프 데이터 준비
    pr_numbers = recent_prs['pr_number_str'].values
    first_review_times = recent_prs['first_review_time'].values
    review_to_merge_times = recent_prs['review_to_merge_time'].values
    
    # 누적 막대 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 첫 번째 단계: 생성부터 첫 리뷰까지
    bars1 = ax.barh(pr_numbers, first_review_times, color='#3498db', alpha=0.8, label='생성 → 첫 리뷰')
    
    # 두 번째 단계: 첫 리뷰부터 병합까지
    bars2 = ax.barh(pr_numbers, review_to_merge_times, left=first_review_times, color='#2ecc71', alpha=0.8, label='첫 리뷰 → 병합')
    
    # 각 단계의 시간을 바 위에 표시
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # 첫 번째 단계 시간
        width1 = bar1.get_width()
        ax.text(width1 / 2, i, f'{width1:.1f}h', ha='center', va='center', color='white', fontweight='bold')
        
        # 두 번째 단계 시간
        width2 = bar2.get_width()
        if width2 > 5:  # 너무 작은 바에는 텍스트 표시 안 함
            ax.text(width1 + width2 / 2, i, f'{width2:.1f}h', ha='center', va='center', color='white', fontweight='bold')
        
        # 전체 시간
        total_width = width1 + width2
        ax.text(total_width + 1, i, f'총 {total_width:.1f}h', va='center')
    
    # 그래프 제목 및 레이블 설정
    ax.set_title('PR 수명 주기 단계별 소요 시간 (최근 병합된 15개 PR)')
    ax.set_xlabel('소요 시간 (시간)')
    ax.set_ylabel('PR 번호')
    ax.legend(loc='upper right')
    
    # Y축 레이블 정렬
    plt.gca().invert_yaxis()  # 최근 PR이 위에 오도록 순서 뒤집기
    
    # 그리드 추가
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_advanced_complexity(df):
    """
    고급 PR 복잡도 지표를 계산합니다.
    
    다음 요소를 고려합니다:
    - 기본 지표: 파일 수, 코드 라인, 리뷰 반복, 코멘트 수
    - 고급 지표: 파일 유형별 가중치, 리뷰 코멘트 깊이, 변경 분산도
    
    Returns:
        복잡도 점수가 추가된 DataFrame
    """
    # 유효한 데이터만 사용
    df_valid = df.copy()
    
    # 1. 기본 지표 정규화 (0-1 범위로)
    # 정규화된 파일 변경 수 (값이 클수록 많은 파일이 변경됨)
    df_valid['normalized_changed_files'] = df_valid['changed_files'] / df_valid['changed_files'].max() if df_valid['changed_files'].max() > 0 else 0
    
    # 정규화된 코드 라인 변경 수 (값이 클수록 많은 라인이 변경됨)
    df_valid['normalized_code_lines'] = (df_valid['additions'] + df_valid['deletions']) / (df_valid['additions'] + df_valid['deletions']).max() if (df_valid['additions'] + df_valid['deletions']).max() > 0 else 0
    
    # 정규화된 리뷰 반복 횟수 (값이 클수록 리뷰가 여러 번 반복됨)
    df_valid['normalized_review_iterations'] = df_valid['review_iterations'] / df_valid['review_iterations'].max() if df_valid['review_iterations'].max() > 0 else 0
    
    # 정규화된 리뷰 코멘트 수 (값이 클수록 많은 코멘트가 달림)
    df_valid['normalized_review_comments'] = df_valid['comments_reviews'] / df_valid['comments_reviews'].max() if df_valid['comments_reviews'].max() > 0 else 0
    
    # 2. 고급 지표 계산
    
    # 2.1 변경 분산도 (파일 수 대비 라인 수 비율의 분산)
    # 파일당 평균 변경 라인 수 계산 (값이 높을수록 변경이 소수 파일에 집중됨)
    df_valid['lines_per_file'] = df_valid.apply(
        lambda row: (row['additions'] + row['deletions']) / row['changed_files'] 
        if row['changed_files'] > 0 else 0, axis=1
    )
    max_lines_per_file = df_valid['lines_per_file'].max()
    
    # 변경 분산도 (값이 높을수록 변경이 여러 파일에 고르게 분산됨)
    df_valid['normalized_change_dispersion'] = 1 - (df_valid['lines_per_file'] / max_lines_per_file) if max_lines_per_file > 0 else 0
    
    # 2.2 리뷰 깊이 (코멘트 수 대비 리뷰 수의 비율)
    # 리뷰당 코멘트 수 계산 (값이 높을수록 리뷰당 코멘트가 많음)
    df_valid['comments_per_review'] = df_valid.apply(
        lambda row: row['comment_count'] / row['review_count'] 
        if row['review_count'] > 0 else 0, axis=1
    )
    max_comments_per_review = df_valid['comments_per_review'].max()
    
    # 정규화된 리뷰 깊이 (값이 높을수록 리뷰 논의가 깊음)
    df_valid['normalized_review_depth'] = df_valid['comments_per_review'] / max_comments_per_review if max_comments_per_review > 0 else 0
    
    # 2.3 PR 크기와 리뷰 반복 횟수의 조합 지표
    # 큰 PR이 여러 번 반복되면 더 복잡 (값이 높을수록 큰 PR이 여러 번 반복됨)
    df_valid['size_iteration_complexity'] = df_valid['normalized_code_lines'] * df_valid['normalized_review_iterations']
    
    # 3. 복합 복잡도 지표 계산 (가중치 적용)
    df_valid['complexity_score'] = (
        0.25 * df_valid['normalized_changed_files'] +      # 파일 수 (25%)
        0.25 * df_valid['normalized_code_lines'] +         # 코드 라인 수 (25%)
        0.15 * df_valid['normalized_review_iterations'] +  # 리뷰 반복 횟수 (15%)
        0.15 * df_valid['normalized_review_comments'] +    # 코멘트 수 (15%)
        0.10 * df_valid['normalized_change_dispersion'] +  # 변경 분산도 (10%)
        0.05 * df_valid['normalized_review_depth'] +       # 리뷰 깊이 (5%)
        0.05 * df_valid['size_iteration_complexity']       # 크기-반복 조합 (5%)
    )
    
    # 복잡도 점수를 0-100 범위로 변환
    df_valid['complexity_score'] = df_valid['complexity_score'] * 100
    
    # 복잡도 범주 정의
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['매우 낮음', '낮음', '중간', '높음', '매우 높음']
    df_valid['complexity_category'] = pd.cut(df_valid['complexity_score'], bins=bins, labels=labels)
    
    return df_valid

def plot_pr_complexity_metrics(df, output_file=None):
    """
    PR 복잡도 지표를 시각화합니다.
    
    복잡도 지표는 다음 요소를 종합적으로 고려합니다:
    - 변경된 파일 수
    - 코드 라인 수 (추가 + 삭제)
    - 리뷰 반복 횟수
    - 코멘트 수
    - 변경 분산도
    - 리뷰 깊이
    - 크기-반복 조합 지표
    
    Args:
        df: PR 데이터가 포함된 DataFrame
        output_file: 출력 파일 경로 (None인 경우 화면에 표시)
    """
    # 유효한 데이터만 사용
    df_valid = df[df['is_merged'] == True].copy()
    
    if len(df_valid) < 5:
        print("경고: 병합된 PR 데이터가 충분하지 않아 복잡도 지표 차트를 생성할 수 없습니다.")
        return
    
    # 고급 복잡도 지표 계산
    df_valid = calculate_advanced_complexity(df_valid)
    
    # 그림 설정
    plt.figure(figsize=(16, 12))
    set_korean_font()
    
    # 1. 복잡도 점수 히스토그램
    plt.subplot(2, 3, 1)
    sns.histplot(df_valid['complexity_score'], bins=20, kde=True)
    plt.title('PR 복잡도 점수 분포')
    plt.xlabel('복잡도 점수')
    plt.ylabel('PR 수')
    
    # 2. 복잡도 범주별 PR 수
    plt.subplot(2, 3, 2)
    category_counts = df_valid['complexity_category'].value_counts().sort_index()
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('복잡도 범주별 PR 수')
    plt.xlabel('복잡도 범주')
    plt.ylabel('PR 수')
    plt.xticks(rotation=45)
    
    # 3. 복잡도 vs 리뷰 시간 산점도
    plt.subplot(2, 3, 3)
    sns.scatterplot(x='complexity_score', y='pr_duration_hours', data=df_valid)
    plt.title('PR 복잡도와 처리 시간의 관계')
    plt.xlabel('복잡도 점수')
    plt.ylabel('PR 처리 시간 (시간)')
    
    # 추세선 추가
    if len(df_valid) > 1:
        z = np.polyfit(df_valid['complexity_score'], df_valid['pr_duration_hours'], 1)
        p = np.poly1d(z)
        plt.plot(df_valid['complexity_score'], p(df_valid['complexity_score']), "r--", alpha=0.8)
    
    # 4. 복잡도 구성 요소 기여도 (상위 15개 PR)
    plt.subplot(2, 3, 4)
    top_complex_prs = df_valid.nlargest(min(15, len(df_valid)), 'complexity_score')
    
    # 각 요소별 기여도 계산
    components = pd.DataFrame({
        'PR': top_complex_prs['pr_number'],
        '파일 수': 0.25 * top_complex_prs['normalized_changed_files'] * 100,
        '코드 라인': 0.25 * top_complex_prs['normalized_code_lines'] * 100,
        '리뷰 반복': 0.15 * top_complex_prs['normalized_review_iterations'] * 100,
        '코멘트 수': 0.15 * top_complex_prs['normalized_review_comments'] * 100,
        '변경 분산도': 0.10 * top_complex_prs['normalized_change_dispersion'] * 100,
        '리뷰 깊이': 0.05 * top_complex_prs['normalized_review_depth'] * 100,
        '크기-반복': 0.05 * top_complex_prs['size_iteration_complexity'] * 100
    })
    
    # 스택 바 차트로 표시
    components_stacked = components.set_index('PR')
    components_stacked.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('복잡도 상위 PR의 구성 요소 기여도')
    plt.xlabel('PR 번호')
    plt.ylabel('기여도 점수')
    plt.xticks(rotation=90)
    plt.legend(title='구성 요소', loc='upper left', bbox_to_anchor=(1, 1))
    
    # 5. 복잡도 요소 간 상관관계 히트맵
    plt.subplot(2, 3, 5)
    complexity_features = [
        'normalized_changed_files', 'normalized_code_lines', 
        'normalized_review_iterations', 'normalized_review_comments',
        'normalized_change_dispersion', 'normalized_review_depth', 
        'size_iteration_complexity'
    ]
    
    # 상관관계 계산을 위한 열 이름 매핑 (히트맵에 표시할 간결한 이름)
    feature_names = {
        'normalized_changed_files': '파일 수',
        'normalized_code_lines': '코드 라인',
        'normalized_review_iterations': '리뷰 반복',
        'normalized_review_comments': '코멘트 수',
        'normalized_change_dispersion': '변경 분산도',
        'normalized_review_depth': '리뷰 깊이',
        'size_iteration_complexity': '크기-반복'
    }
    
    correlation = df_valid[complexity_features].corr()
    
    # 상관관계 히트맵에 표시할 열 이름 변경
    correlation.index = [feature_names[col] for col in correlation.index]
    correlation.columns = [feature_names[col] for col in correlation.columns]
    
    # 상관관계 히트맵
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('복잡도 요소 간 상관관계')
    
    # 6. 복잡도 요소별 중요도 (가중치)
    plt.subplot(2, 3, 6)
    weights = {
        '파일 수': 25,
        '코드 라인': 25,
        '리뷰 반복': 15,
        '코멘트 수': 15,
        '변경 분산도': 10,
        '리뷰 깊이': 5,
        '크기-반복': 5
    }
    
    # 가중치 막대 그래프
    plt.bar(weights.keys(), weights.values())
    plt.title('복잡도 요소별 가중치')
    plt.xlabel('복잡도 요소')
    plt.ylabel('가중치 (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 파일로 저장하거나 화면에 표시
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
    
    # 새로 추가된 PR 수명 주기 단계별 소요 시간 분석 차트
    plot_pr_lifecycle_stages(df,
        None if args.show else os.path.join(args.output_dir, f"pr_lifecycle_stages{date_suffix}.png"))
    
    # 새로 추가된 PR 복잡도 지표 차트
    plot_pr_complexity_metrics(df,
        None if args.show else os.path.join(args.output_dir, f"pr_complexity_metrics{date_suffix}.png"))
    
    if not args.show:
        print(f"모든 차트가 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
