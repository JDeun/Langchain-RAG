import re
from datetime import datetime, timedelta

def parse_date(date_string):
    if date_string is None:
        return None
    
    # 기존의 날짜 형식 리스트
    date_formats = [
        "%Y.%m.%d. %p %I:%M",
        "%Y.%m.%d.",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y년 %m월 %d일 %H시 %M분",
    ]
    
    # 기존 형식으로 파싱 시도
    for fmt in date_formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass

    # 정규표현식을 사용한 날짜 추출
    patterns = [
        r"(\d{4})[.-](\d{1,2})[.-](\d{1,2})",  # YYYY-MM-DD or YYYY.MM.DD
        r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",  # YYYY년 MM월 DD일
        r"(\d{2})\.(\d{2})\.(\d{2})",  # YY.MM.DD
        r"(\d{1,2})일\s*전",  # N일 전
        r"(\d{1,2})시간\s*전",  # N시간 전
        r"(\d{1,2})분\s*전",  # N분 전
    ]

    for pattern in patterns:
        match = re.search(pattern, date_string)
        if match:
            try:
                if "일 전" in date_string:
                    days_ago = int(match.group(1))
                    return datetime.now() - timedelta(days=days_ago)
                elif "시간 전" in date_string:
                    hours_ago = int(match.group(1))
                    return datetime.now() - timedelta(hours=hours_ago)
                elif "분 전" in date_string:
                    minutes_ago = int(match.group(1))
                    return datetime.now() - timedelta(minutes=minutes_ago)
                else:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))
                    if year < 100:  # YY.MM.DD 형식 처리
                        year += 2000
                    return datetime(year, month, day)
            except ValueError:
                pass

    print(f"날짜 파싱 실패: {date_string}")
    return None

def is_recent_news(date, days=7):
    if date is None:
        return False
    return datetime.now() - timedelta(days=days) <= date <= datetime.now()