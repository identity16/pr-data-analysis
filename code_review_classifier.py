#!/usr/bin/env python3
import os
import json
import requests
import logging
import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_directory, f"classification_log_{current_time}.log")

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("code_review_classifier")

# OpenAI API 상수
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# 코드 리뷰 카테고리 정의 및 설명
REVIEW_CATEGORIES_WITH_DESCRIPTIONS = {
    "버그 수정 제안": "코드의 오류나 버그를 지적하고 수정 방법을 제안하는 코멘트",
    "코드 품질 개선": "가독성, 유지보수성, 코드 구조 등 코드 품질 향상을 위한 제안",
    "성능 최적화": "실행 시간, 메모리 사용량 등 성능 개선을 위한 제안",
    "보안 이슈": "보안 취약점이나 잠재적인 보안 문제를 지적하는 코멘트",
    "기능 제안": "새로운 기능 추가나 기존 기능 확장에 대한 제안",
    "문서화 요청": "주석, 문서, README 등의 개선이나 추가를 요청하는 코멘트",
    "테스트 관련": "테스트 코드 추가, 테스트 케이스 개선 등에 관한 코멘트",
    "스타일 가이드 준수": "코딩 스타일, 네이밍 컨벤션 등 스타일 가이드 준수에 관한 코멘트",
    "아키텍처 개선": "코드 구조, 설계 패턴, 아키텍처 관련 개선 제안",
    "일반 대화": "코드 리뷰와 직접 관련이 없는 인사, 감사, 질문, 잡담 등의 일반적인 대화",
    "기타": "위 카테고리에 명확하게 속하지 않는 코멘트 (다른 카테고리에 속할 가능성이 없는 경우에만 사용)"
}

# 카테고리 리스트
REVIEW_CATEGORIES = list(REVIEW_CATEGORIES_WITH_DESCRIPTIONS.keys())

# 신뢰도 임계값 - 이 값보다 낮은 신뢰도로 '기타'로 분류된 경우 재분류 시도
CONFIDENCE_THRESHOLD = 0.7

# 일반 대화 예시 - 분류 정확도 향상을 위한 예시
GENERAL_CONVERSATION_EXAMPLES = [
    "감사합니다!",
    "확인했습니다.",
    "네, 알겠습니다.",
    "좋은 의견 감사합니다.",
    "다음 PR에 반영하겠습니다.",
    "이 부분은 나중에 다시 논의해 보죠.",
    "오늘 회의에서 이야기해 볼까요?",
    "주말 잘 보내세요!",
    "질문이 있으면 언제든 물어보세요.",
    "이 PR은 언제쯤 머지될 예정인가요?"
]

def classify_review_comment(comment_text: str, comment_id: str = None) -> Dict[str, Any]:
    """
    LLM을 사용하여 코드 리뷰 코멘트를 분류합니다.
    
    매개변수:
    - comment_text: 분류할 코드 리뷰 코멘트 텍스트
    - comment_id: 코멘트 식별자 (로깅용)
    
    반환값:
    - 분류 결과를 포함하는 딕셔너리
    """
    comment_id = comment_id or "unknown_id"
    logger.info(f"[{comment_id}] 코멘트 분류 시작: {comment_text[:100]}{'...' if len(comment_text) > 100 else ''}")
    
    if not OPENAI_API_KEY:
        logger.warning("경고: OPENAI_API_KEY가 설정되지 않았습니다. 코드 리뷰 분류를 건너뜁니다.")
        return {"category": "미분류", "confidence": 0.0}
    
    # 카테고리 설명 포맷팅
    category_descriptions = "\n".join([f"- {category}: {description}" for category, description in REVIEW_CATEGORIES_WITH_DESCRIPTIONS.items()])
    
    # 일반 대화 예시 포맷팅
    conversation_examples = "\n".join([f"- \"{example}\"" for example in GENERAL_CONVERSATION_EXAMPLES])
    
    # 프롬프트 구성
    prompt = f"""
다음 GitHub 코드 리뷰 코멘트를 분석하고 가장 적합한 카테고리를 선택해주세요:

코멘트: "{comment_text}"

다음 카테고리 중 하나를 선택하세요:
{category_descriptions}

특별 지침:
1. '일반 대화' 카테고리는 코드 리뷰와 직접 관련이 없는 대화에 사용합니다. 다음은 '일반 대화'의 예시입니다:
{conversation_examples}

2. '기타' 카테고리는 코드 리뷰와 관련은 있지만 다른 어떤 카테고리에도 명확하게 속하지 않을 때만 사용하세요.

3. 가능한 한 구체적인 카테고리를 선택하고, 여러 카테고리에 해당할 경우 가장 핵심적인 목적에 맞는 카테고리를 선택하세요.

응답 형식:
{{
  "category": "선택한 카테고리",
  "confidence": 신뢰도(0.0~1.0),
  "reasoning": "선택 이유에 대한 간략한 설명"
}}
"""
    
    # API 요청 헤더
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # API 요청 데이터
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "당신은 코드 리뷰 분석 전문가입니다. 코드 리뷰 코멘트를 분석하고 적절한 카테고리로 분류해주세요. 코드 리뷰와 관련 없는 일반 대화는 '일반 대화' 카테고리로, '기타' 카테고리는 최후의 수단으로만 사용하세요."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,  # 더 결정적인 응답을 위해 온도 낮춤
        "max_tokens": 500
    }
    
    try:
        # API 요청 보내기
        logger.debug(f"[{comment_id}] OpenAI API 요청 시작")
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        logger.debug(f"[{comment_id}] OpenAI API 응답 수신 완료")
        
        # 응답 파싱
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        logger.debug(f"[{comment_id}] 응답 메시지: {assistant_message}")
        
        # JSON 형식 추출
        try:
            # JSON 부분 추출 시도
            json_start = assistant_message.find("{")
            json_end = assistant_message.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = assistant_message[json_start:json_end]
                classification = json.loads(json_str)
            else:
                # JSON 형식이 아닌 경우 전체 텍스트를 파싱
                classification = json.loads(assistant_message)
            
            logger.info(f"[{comment_id}] 초기 분류 결과: 카테고리={classification.get('category', '미분류')}, 신뢰도={classification.get('confidence', 0.0)}")
            logger.info(f"[{comment_id}] 분류 이유: {classification.get('reasoning', '이유 없음')}")
            
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본값 반환
            logger.error(f"[{comment_id}] 경고: LLM 응답을 JSON으로 파싱할 수 없습니다: {assistant_message}")
            return {"category": "미분류", "confidence": 0.0, "reasoning": "파싱 오류"}
        
        # '기타' 카테고리이고 신뢰도가 낮은 경우 재분류 시도
        if classification.get("category") == "기타" and classification.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            logger.info(f"[{comment_id}] '기타' 카테고리 신뢰도({classification.get('confidence', 0.0)})가 임계값({CONFIDENCE_THRESHOLD}) 미만, 재분류 시도")
            
            # 재분류를 위한 프롬프트 - '기타' 카테고리 제외하고 '일반 대화' 카테고리 강조
            retry_categories = {k: v for k, v in REVIEW_CATEGORIES_WITH_DESCRIPTIONS.items() if k != "기타"}
            retry_category_descriptions = "\n".join([f"- {category}: {description}" for category, description in retry_categories.items()])
            
            retry_prompt = f"""
다음 GitHub 코드 리뷰 코멘트를 다시 분석하고 가장 적합한 카테고리를 선택해주세요:

코멘트: "{comment_text}"

다음 카테고리 중 하나를 선택하세요 ('기타' 카테고리는 제외됨):
{retry_category_descriptions}

특별 지침:
1. 코드 리뷰와 직접 관련이 없는 인사, 감사, 질문, 잡담 등은 '일반 대화' 카테고리로 분류하세요. 다음은 '일반 대화'의 예시입니다:
{conversation_examples}

2. 가장 적합한 카테고리를 선택하세요. 여러 카테고리에 해당할 경우 가장 핵심적인 목적에 맞는 카테고리를 선택하세요.

응답 형식:
{{
  "category": "선택한 카테고리",
  "confidence": 신뢰도(0.0~1.0),
  "reasoning": "선택 이유에 대한 간략한 설명"
}}
"""
            # 재분류 API 요청 데이터
            retry_data = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "당신은 코드 리뷰 분석 전문가입니다. 코드 리뷰 코멘트를 분석하고 적절한 카테고리로 분류해주세요. 코드 리뷰와 관련 없는 일반 대화는 '일반 대화' 카테고리로 분류하세요."},
                    {"role": "user", "content": retry_prompt}
                ],
                "temperature": 0.1,  # 더 결정적인 응답을 위해 온도 더 낮춤
                "max_tokens": 500
            }
            
            # 재분류 API 요청 보내기
            logger.debug(f"[{comment_id}] 재분류 OpenAI API 요청 시작")
            retry_response = requests.post(OPENAI_API_URL, headers=headers, json=retry_data)
            retry_response.raise_for_status()
            logger.debug(f"[{comment_id}] 재분류 OpenAI API 응답 수신 완료")
            
            # 재분류 응답 파싱
            retry_result = retry_response.json()
            retry_assistant_message = retry_result["choices"][0]["message"]["content"]
            logger.debug(f"[{comment_id}] 재분류 응답 메시지: {retry_assistant_message}")
            
            try:
                # JSON 부분 추출 시도
                json_start = retry_assistant_message.find("{")
                json_end = retry_assistant_message.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = retry_assistant_message[json_start:json_end]
                    retry_classification = json.loads(json_str)
                else:
                    # JSON 형식이 아닌 경우 전체 텍스트를 파싱
                    retry_classification = json.loads(retry_assistant_message)
                
                logger.info(f"[{comment_id}] 재분류 결과: 카테고리={retry_classification.get('category', '미분류')}, 신뢰도={retry_classification.get('confidence', 0.0)}")
                logger.info(f"[{comment_id}] 재분류 이유: {retry_classification.get('reasoning', '이유 없음')}")
                
                # 재분류 결과가 더 높은 신뢰도를 가지면 사용
                if retry_classification.get("confidence", 0.0) > classification.get("confidence", 0.0):
                    logger.info(f"[{comment_id}] 재분류 결과 채택: 신뢰도 {classification.get('confidence', 0.0)} -> {retry_classification.get('confidence', 0.0)}")
                    classification = retry_classification
                    classification["was_reclassified"] = True
                else:
                    logger.info(f"[{comment_id}] 재분류 결과 기각: 신뢰도 {retry_classification.get('confidence', 0.0)}가 원래 신뢰도 {classification.get('confidence', 0.0)}보다 낮거나 같음")
            except json.JSONDecodeError:
                # 재분류 실패 시 원래 분류 유지
                logger.error(f"[{comment_id}] 재분류 응답을 JSON으로 파싱할 수 없습니다: {retry_assistant_message}")
        
        # 최종 분류 결과 로깅
        logger.info(f"[{comment_id}] 최종 분류 결과: 카테고리={classification.get('category', '미분류')}, 신뢰도={classification.get('confidence', 0.0)}, 재분류={classification.get('was_reclassified', False)}")
        
        # 분류 결과를 JSON 파일로도 저장
        save_classification_to_json(comment_id, comment_text, classification)
        
        return classification
    
    except requests.exceptions.RequestException as e:
        logger.error(f"[{comment_id}] API 요청 오류: {e}")
        return {"category": "미분류", "confidence": 0.0, "reasoning": f"API 오류: {str(e)}"}
    except Exception as e:
        logger.error(f"[{comment_id}] 예상치 못한 오류: {e}")
        return {"category": "미분류", "confidence": 0.0, "reasoning": f"오류: {str(e)}"}

def save_classification_to_json(comment_id: str, comment_text: str, classification: Dict[str, Any]):
    """
    분류 결과를 JSON 파일로 저장합니다.
    
    매개변수:
    - comment_id: 코멘트 식별자
    - comment_text: 분류된 코멘트 텍스트
    - classification: 분류 결과
    """
    json_directory = os.path.join(log_directory, "classifications")
    os.makedirs(json_directory, exist_ok=True)
    
    # 결과를 저장할 JSON 파일 경로
    json_file = os.path.join(json_directory, f"classification_{comment_id}_{current_time}.json")
    
    # 저장할 데이터 구성
    data = {
        "comment_id": comment_id,
        "comment_text": comment_text,
        "classification": classification,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # JSON 파일로 저장
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"[{comment_id}] 분류 결과를 JSON 파일로 저장: {json_file}")
    except Exception as e:
        logger.error(f"[{comment_id}] JSON 파일 저장 오류: {e}")

def classify_thread_comments(thread_comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    동일 스레드의 코멘트들을 묶어서 분류합니다.
    
    매개변수:
    - thread_comments: 동일 스레드에 속한 코멘트 목록
    
    반환값:
    - 스레드 전체에 대한 분류 결과를 포함하는 딕셔너리
    """
    if not thread_comments:
        return {"category": "미분류", "confidence": 0.0}
    
    # 스레드의 모든 코멘트를 하나의 텍스트로 결합
    thread_text = "\n\n".join([f"[{comment['user']['login']}]: {comment['body']}" for comment in thread_comments])
    thread_id = f"thread_{thread_comments[0]['id']}"
    
    logger.info(f"[{thread_id}] 스레드 분류 시작: {len(thread_comments)}개 코멘트")
    
    # 결합된 텍스트를 분류
    classification = classify_review_comment(thread_text, thread_id)
    
    return classification

def organize_comments_by_thread(comments: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    코멘트를 스레드별로 구성합니다.
    
    매개변수:
    - comments: PR 코멘트 목록
    
    반환값:
    - 스레드별로 구성된 코멘트 목록의 목록
    """
    # 코멘트 ID를 키로 하는 딕셔너리 생성
    comment_dict = {str(comment["id"]): comment for comment in comments}
    
    # 스레드 루트 코멘트 (in_reply_to_id가 없는 코멘트) 식별
    root_comments = []
    reply_comments = []
    
    for comment in comments:
        if comment.get("in_reply_to_id") is None:
            root_comments.append(comment)
        else:
            reply_comments.append(comment)
    
    # 스레드별로 코멘트 구성
    threads = []
    
    # 루트 코멘트를 기준으로 스레드 구성
    for root in root_comments:
        thread = [root]
        root_id = str(root["id"])
        
        # 이 루트에 대한 답글 찾기
        for reply in reply_comments:
            if str(reply.get("in_reply_to_id")) == root_id:
                thread.append(reply)
        
        threads.append(thread)
    
    # 루트가 없는 답글 처리 (API 응답에 루트 코멘트가 포함되지 않은 경우)
    remaining_replies = []
    for reply in reply_comments:
        reply_to_id = str(reply.get("in_reply_to_id"))
        if reply_to_id not in comment_dict:
            remaining_replies.append(reply)
    
    # 남은 답글을 개별 스레드로 처리
    for reply in remaining_replies:
        threads.append([reply])
    
    logger.info(f"총 {len(comments)}개 코멘트를 {len(threads)}개 스레드로 구성했습니다.")
    return threads

def classify_pr_reviews(reviews: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    PR의 모든 리뷰와 코멘트를 분류하고 통계를 계산합니다.
    
    매개변수:
    - reviews: PR 리뷰 목록
    - comments: PR 코멘트 목록
    
    반환값:
    - 분류 결과와 통계를 포함하는 딕셔너리
    """
    logger.info(f"PR 분류 시작: 리뷰 {len(reviews)}개, 코멘트 {len(comments)}개")
    
    # 분류 결과를 저장할 리스트
    classifications = []
    
    # 리뷰 분류
    for review in reviews:
        if review.get("body") and review.get("body").strip():
            review_id = str(review["id"])
            logger.info(f"리뷰 분류 시작 (ID: {review_id}, 사용자: {review['user']['login']})")
            classification = classify_review_comment(review["body"], f"review_{review_id}")
            classifications.append({
                "type": "review",
                "id": review_id,
                "user": review["user"]["login"],
                "text": review["body"],
                "classification": classification
            })
    
    # 코멘트를 스레드별로 구성
    comment_threads = organize_comments_by_thread(comments)
    
    # 스레드별 분류
    for thread in comment_threads:
        if not thread:
            continue
            
        thread_id = str(thread[0]["id"])
        
        # 스레드에 코멘트가 하나만 있는 경우 개별 분류
        if len(thread) == 1:
            comment = thread[0]
            if comment.get("body") and comment.get("body").strip():
                logger.info(f"단일 코멘트 분류 시작 (ID: {thread_id}, 사용자: {comment['user']['login']})")
                classification = classify_review_comment(comment["body"], f"comment_{thread_id}")
                classifications.append({
                    "type": "comment",
                    "id": thread_id,
                    "user": comment["user"]["login"],
                    "text": comment["body"],
                    "classification": classification,
                    "thread_size": 1
                })
        # 스레드에 여러 코멘트가 있는 경우 스레드 전체 분류
        else:
            logger.info(f"스레드 분류 시작 (루트 ID: {thread_id}, 코멘트 수: {len(thread)})")
            classification = classify_thread_comments(thread)
            
            # 스레드의 모든 사용자 목록
            users = list(set(comment["user"]["login"] for comment in thread))
            
            # 스레드의 모든 텍스트 결합
            thread_text = "\n\n".join([f"[{comment['user']['login']}]: {comment['body']}" for comment in thread])
            
            classifications.append({
                "type": "thread",
                "id": thread_id,
                "users": users,
                "text": thread_text,
                "classification": classification,
                "thread_size": len(thread)
            })
    
    # 카테고리별 통계 계산
    category_counts = {category: 0 for category in REVIEW_CATEGORIES}
    category_counts["미분류"] = 0
    reclassified_count = 0
    thread_count = 0
    
    for item in classifications:
        category = item["classification"].get("category", "미분류")
        category_counts[category] = category_counts.get(category, 0) + 1
        if item["classification"].get("was_reclassified", False):
            reclassified_count += 1
        if item["type"] == "thread" and item.get("thread_size", 0) > 1:
            thread_count += 1
    
    # 통계 로깅
    logger.info("분류 통계:")
    for category, count in category_counts.items():
        logger.info(f"  - {category}: {count}개")
    logger.info(f"재분류된 항목: {reclassified_count}개")
    logger.info(f"스레드로 분류된 항목: {thread_count}개")
    
    # 전체 분류 결과를 JSON 파일로 저장
    save_all_classifications_to_json(classifications, category_counts)
    
    # 결과 반환
    return {
        "classifications": classifications,
        "category_counts": category_counts,
        "total_classified_items": len(classifications),
        "reclassified_count": reclassified_count,
        "thread_count": thread_count
    }

def save_all_classifications_to_json(classifications: List[Dict[str, Any]], category_counts: Dict[str, int]):
    """
    모든 분류 결과를 하나의 JSON 파일로 저장합니다.
    
    매개변수:
    - classifications: 모든 분류 결과 목록
    - category_counts: 카테고리별 개수
    """
    json_directory = os.path.join(log_directory, "summary")
    os.makedirs(json_directory, exist_ok=True)
    
    # 결과를 저장할 JSON 파일 경로
    json_file = os.path.join(json_directory, f"classification_summary_{current_time}.json")
    
    # 저장할 데이터 구성
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_items": len(classifications),
        "category_counts": category_counts,
        "classifications": classifications
    }
    
    # JSON 파일로 저장
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"전체 분류 결과를 JSON 파일로 저장: {json_file}")
    except Exception as e:
        logger.error(f"전체 JSON 파일 저장 오류: {e}")

def get_review_classification_metrics(reviews: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    PR의 리뷰와 코멘트를 분류하고 카테고리별 개수를 반환합니다.
    이 함수는 github-pr-metrics.py에서 호출하기 위한 간소화된 인터페이스입니다.
    
    매개변수:
    - reviews: PR 리뷰 목록
    - comments: PR 코멘트 목록
    
    반환값:
    - 카테고리별 개수를 포함하는 딕셔너리
    """
    if not OPENAI_API_KEY:
        # API 키가 없으면 빈 결과 반환
        logger.warning("OPENAI_API_KEY가 설정되지 않았습니다. 빈 분류 결과를 반환합니다.")
        return {f"review_category_{category.lower().replace(' ', '_')}": 0 for category in REVIEW_CATEGORIES}
    
    # 분류 수행
    logger.info("PR 리뷰 분류 메트릭 계산 시작")
    classification_results = classify_pr_reviews(reviews, comments)
    
    # 결과 형식 변환 (github-pr-metrics.py에서 사용하기 쉽게)
    metrics = {}
    for category, count in classification_results["category_counts"].items():
        # 카테고리 이름을 스네이크 케이스로 변환
        category_key = f"review_category_{category.lower().replace(' ', '_')}"
        metrics[category_key] = count
    
    # 재분류 정보 추가
    metrics["review_reclassified_count"] = classification_results.get("reclassified_count", 0)
    
    # 스레드 정보 추가
    metrics["review_thread_count"] = classification_results.get("thread_count", 0)
    metrics["review_total_items"] = classification_results.get("total_classified_items", 0)
    
    logger.info("PR 리뷰 분류 메트릭 계산 완료")
    return metrics

def analyze_classification_results():
    """
    로그 디렉토리에서 분류 결과를 분석하고 요약 보고서를 생성합니다.
    """
    summary_dir = os.path.join(log_directory, "summary")
    if not os.path.exists(summary_dir):
        logger.error("요약 디렉토리가 존재하지 않습니다.")
        return
    
    # 가장 최근의 요약 파일 찾기
    summary_files = [f for f in os.listdir(summary_dir) if f.startswith("classification_summary_")]
    if not summary_files:
        logger.error("요약 파일이 존재하지 않습니다.")
        return
    
    latest_summary = max(summary_files)
    summary_path = os.path.join(summary_dir, latest_summary)
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 카테고리별 분석
        other_items = [item for item in data["classifications"] if item["classification"].get("category") == "기타"]
        conversation_items = [item for item in data["classifications"] if item["classification"].get("category") == "일반 대화"]
        
        # 분석 보고서 생성
        report_path = os.path.join(log_directory, f"analysis_report_{current_time}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 코드 리뷰 분류 분석 보고서 ===\n\n")
            f.write(f"분석 시간: {datetime.datetime.now().isoformat()}\n")
            f.write(f"데이터 소스: {summary_path}\n\n")
            
            f.write("== 카테고리별 통계 ==\n")
            for category, count in data["category_counts"].items():
                percentage = (count / data["total_items"]) * 100 if data["total_items"] > 0 else 0
                f.write(f"{category}: {count}개 ({percentage:.1f}%)\n")
            
            f.write(f"\n== '일반 대화' 카테고리 분석 ({len(conversation_items)}개) ==\n")
            for i, item in enumerate(conversation_items, 1):
                f.write(f"\n{i}. ID: {item['id']} (타입: {item['type']}, 사용자: {item['user']})\n")
                f.write(f"   텍스트: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}\n")
                f.write(f"   신뢰도: {item['classification'].get('confidence', 0.0)}\n")
                f.write(f"   이유: {item['classification'].get('reasoning', '이유 없음')}\n")
                f.write(f"   재분류: {'예' if item['classification'].get('was_reclassified', False) else '아니오'}\n")
            
            f.write(f"\n== '기타' 카테고리 분석 ({len(other_items)}개) ==\n")
            for i, item in enumerate(other_items, 1):
                f.write(f"\n{i}. ID: {item['id']} (타입: {item['type']}, 사용자: {item['user']})\n")
                f.write(f"   텍스트: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}\n")
                f.write(f"   신뢰도: {item['classification'].get('confidence', 0.0)}\n")
                f.write(f"   이유: {item['classification'].get('reasoning', '이유 없음')}\n")
                f.write(f"   재분류 시도: {'예' if item['classification'].get('was_reclassified', False) else '아니오'}\n")
        
        logger.info(f"분석 보고서가 생성되었습니다: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"분석 보고서 생성 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    # 테스트 코드
    logger.info("테스트 분류 시작")
    
    test_comment = "이 부분에서 성능 이슈가 있을 수 있습니다. O(n^2) 대신 해시맵을 사용하여 O(n)으로 최적화하는 것이 좋겠습니다."
    result = classify_review_comment(test_comment, "test_performance")
    print(f"분류 결과: {result}")
    
    # 기타로 분류될 가능성이 있는 모호한 코멘트 테스트
    ambiguous_comment = "이 부분을 확인해 주세요."
    result = classify_review_comment(ambiguous_comment, "test_ambiguous")
    print(f"모호한 코멘트 분류 결과: {result}")
    
    # 일반 대화 테스트
    conversation_comment = "감사합니다! 수정사항 반영했습니다."
    result = classify_review_comment(conversation_comment, "test_conversation")
    print(f"일반 대화 분류 결과: {result}")
    
    # 분석 보고서 생성
    report_path = analyze_classification_results()
    if report_path:
        print(f"분석 보고서가 생성되었습니다: {report_path}") 