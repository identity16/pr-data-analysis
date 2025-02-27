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

def load_pr_data(file_path):
    """CSV 파일에서 PR 메트릭 데이터를 로드합니다."""
    df = pd.read_csv(file_path)
    
    # 날짜 문자열을 datetime 객체로 변환
    date_columns = [col for col in df.columns if col.endswith('_at')]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
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
    plt.title('PR 기간 분포 (시간)')
    plt.xlabel('기간 (시간)')
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
    plt.title('PR 크기 대 기간')
    plt.xlabel('PR 크기 (변경된 라인 수)')
    plt.ylabel('기간 (시간)')
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
    plt.figure(figsize=(12, 8))
    
    # 모든 리뷰어 추출
    all_reviewers = []
    for reviewers_list in df['reviewers'].dropna():
        # 문자열 형태의 리스트를 실제 리스트로 변환
        if isinstance(reviewers_list, str):
            try:
                reviewers_list = eval(reviewers_list)
            except:
                reviewers_list = []
        all_reviewers.extend(reviewers_list)
    
    if not all_reviewers:
        print("리뷰어 데이터가 없습니다")
        return
    
    # 리뷰어별 리뷰 수 계산
    reviewer_counts = pd.Series(all_reviewers).value_counts()
    
    # 가독성을 위해 상위 15명의 리뷰어 선택 (또는 15명보다 적으면 전체)
    top_n = min(15, len(reviewer_counts))
    top_reviewers = reviewer_counts.head(top_n)
    
    # 그래프 작성
    sns.barplot(y=top_reviewers.index, x=top_reviewers.values)
    plt.title(f'리뷰어별 리뷰 부하 (상위 {top_n}명)')
    plt.xlabel('리뷰 수')
    plt.ylabel('리뷰어')
    plt.grid(True, alpha=0.3)
    
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
    plt.figure(figsize=(12, 6))
    
    # 월/년별로 그룹화
    if 'created_yearmonth' not in df.columns and 'created_at' in df.columns:
        df['created_yearmonth'] = df['created_at'].dt.strftime('%Y-%m')
    
    if 'created_yearmonth' not in df.columns:
        print("월별 생성 시간 열이 없습니다")
        return
    
    monthly_counts = df.groupby('created_yearmonth').size()
    
    # 그래프 작성
    monthly_counts.plot(kind='bar')
    plt.title('월별 생성된 PR')
    plt.xlabel('월')
    plt.ylabel('PR 수')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
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
    # 명령행 인수 구문 분석
    parser = argparse.ArgumentParser(description="GitHub PR 메트릭 시각화")
    parser.add_argument("input_file", help="PR 메트릭이 포함된 입력 CSV 파일")
    parser.add_argument("--output-dir", default="charts", help="차트 출력 디렉토리")
    parser.add_argument("--show", action="store_true", help="그래프를 화면에 표시합니다")
    
    args = parser.parse_args()
    
    # 존재하지 않는 경우 출력 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 데이터 로드
    df = load_pr_data(args.input_file)
    
    # 차트 생성
    print("차트 생성 중...")
    
    # 각 시각화 함수에 대해 파일 이름 생성
    if args.show:
        output_dir = None
    else:
        output_dir = args.output_dir
    
    plot_pr_duration_histogram(df, 
        None if args.show else os.path.join(args.output_dir, "pr_duration_histogram.png"))
    
    plot_pr_size_vs_duration(df, 
        None if args.show else os.path.join(args.output_dir, "pr_size_vs_duration.png"))
    
    plot_review_time_trend(df, 
        None if args.show else os.path.join(args.output_dir, "review_time_trend.png"))
    
    plot_review_load_by_reviewer(df, 
        None if args.show else os.path.join(args.output_dir, "review_load_by_reviewer.png"))
    
    # 새로 추가된 승인 리뷰어 차트
    plot_approved_reviewers(df, 
        None if args.show else os.path.join(args.output_dir, "approved_reviewers.png"))
    
    # 새로 추가된 리뷰어별 승인 비율 차트
    plot_approval_ratio_by_reviewer(df, 
        None if args.show else os.path.join(args.output_dir, "approval_ratio_by_reviewer.png"))
    
    plot_pr_creation_over_time(df, 
        None if args.show else os.path.join(args.output_dir, "pr_creation_over_time.png"))
    
    plot_pr_outcome_by_size(df, 
        None if args.show else os.path.join(args.output_dir, "pr_outcome_by_size.png"))
    
    plot_review_iterations_by_size(df, 
        None if args.show else os.path.join(args.output_dir, "review_iterations_by_size.png"))
    
    plot_reviewers_per_pr(df, 
        None if args.show else os.path.join(args.output_dir, "reviewers_per_pr.png"))
    
    plot_pr_throughput_over_time(df, 
        None if args.show else os.path.join(args.output_dir, "pr_throughput_over_time.png"))
    
    plot_review_comments_per_pr_size(df, 
        None if args.show else os.path.join(args.output_dir, "review_comments_per_pr_size.png"))
    
    plot_pr_size_distribution(df, 
        None if args.show else os.path.join(args.output_dir, "pr_size_distribution.png"))
    
    if not args.show:
        print(f"차트가 {args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
