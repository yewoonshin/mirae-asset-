from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

from recommend_products import (
    print_result,
    recommend_products,
)


# =========================================================
# HyperCLOVA X 설정
# =========================================================

CLOVA_BASE_URL = os.getenv(
    "CLOVA_STUDIO_BASE_URL",
    "https://clovastudio.stream.ntruss.com/v1/openai",
)

CLOVA_MODEL = os.getenv(
    "CLOVA_STUDIO_MODEL",
    "HCX-005",
)


# =========================================================
# Function Calling 스키마
# =========================================================

PARSE_TOOL = {
    "type": "function",
    "function": {
        "name": "parse_recommendation_request",
        "description": (
            "사용자의 자연어 연금 상품 추천 요청을 "
            "deterministic 추천 엔진이 사용할 구조화 조건으로 변환한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "recommendation",
                        "unsupported",
                    ],
                    "description": (
                        "현재 상품 DB로 추천할 수 있으면 recommendation, "
                        "현재 범위를 벗어나면 unsupported."
                    ),
                },
                "account_type": {
                    "type": "string",
                    "enum": [
                        "IRP",
                        "퇴직연금",
                        "연금저축",
                    ],
                    "description": (
                        "사용자가 명시한 연금계좌 유형. "
                        "명시되지 않았거나 '연금'처럼 모호하면 생략한다."
                    ),
                },
                "product_type": {
                    "type": "string",
                    "enum": [
                        "채권형",
                        "주식형",
                    ],
                    "description": (
                        "상품유형. 사용자가 직접 말했거나 "
                        "단기채/국공채/회사채/주식/배당주 등으로 "
                        "명확하게 추론 가능한 경우에만 설정한다."
                    ),
                },
                "risk_grade_min": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "description": (
                        "허용 위험등급의 최소 숫자. "
                        "한국 펀드 위험등급은 1이 가장 위험하고 "
                        "6이 가장 낮은 위험이다."
                    ),
                },
                "risk_grade_max": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "description": (
                        "허용 위험등급의 최대 숫자."
                    ),
                },
                "online_only": {
                    "type": "boolean",
                    "description": (
                        "온라인/모바일/비대면 상품만 원하면 true."
                    ),
                },
                "preferred_channel": {
                    "type": "string",
                    "enum": [
                        "online",
                        "online_super",
                        "offline",
                        "default_option",
                    ],
                    "description": (
                        "특정 판매채널을 명시한 경우에만 설정한다."
                    ),
                },
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "펀드명 적합도에 활용할 핵심 전략 키워드. "
                        "예: 단기, 국공채, 고배당, ESG, 미국."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": (
                        "추천 개수. 별도 언급 없으면 5."
                    ),
                },
                "needs_clarification": {
                    "type": "boolean",
                    "description": (
                        "실제 추천을 위해 사용자에게 한 가지를 "
                        "더 물어봐야 하면 true."
                    ),
                },
                "clarifying_question": {
                    "type": "string",
                    "description": (
                        "needs_clarification=true일 때 사용자에게 "
                        "보여줄 짧은 추가 질문."
                    ),
                },
                "unsupported_reason": {
                    "type": "string",
                    "description": (
                        "intent=unsupported일 때 현재 DB 범위를 "
                        "벗어나는 이유."
                    ),
                },
                "interpretation": {
                    "type": "string",
                    "description": (
                        "사용자 조건을 어떻게 해석했는지 한 문장으로 요약."
                    ),
                },
            },
            "required": [
                "intent",
                "online_only",
                "top_k",
                "needs_clarification",
                "clarifying_question",
                "unsupported_reason",
                "interpretation",
            ],
            "additionalProperties": False,
        },
    },
}


SYSTEM_PROMPT = """
당신은 연금 상품 추천 요청을 구조화하는 HyperCLOVA X 라우터입니다.

중요 원칙:
1. 당신은 상품을 직접 추천하거나 순위를 매기지 않습니다.
2. 자연어를 조건으로만 변환합니다.
3. 실제 상품 필터링/정렬은 별도의 deterministic Python 엔진이 수행합니다.
4. 모르는 조건을 임의로 만들어내지 않습니다.

현재 DB의 추천 대상:
- 투자설명서에서 추출한 공모 펀드
- 연금저축 / 퇴직연금 / IRP 관련 class
- 채권형 / 주식형 등
- 위험등급, 판매채널, 보수/비용을 기반으로 추천

계좌 유형 규칙:
- "IRP", "개인형퇴직연금", "개인퇴직계좌" -> account_type="IRP"
- "퇴직연금", "DC", "DB" -> account_type="퇴직연금"
- "연금저축", "개인연금" -> account_type="연금저축"
- 사용자가 단순히 "연금"이라고만 하면 계좌가 모호하므로
  needs_clarification=true로 하고
  "IRP, 퇴직연금, 연금저축 중 어떤 계좌인가요?"라고 질문합니다.
- 실제 class 추천에는 계좌유형이 중요하므로 계좌가 전혀 언급되지 않은
  상품 추천 요청도 원칙적으로 계좌유형을 한 번 확인합니다.

상품 유형 규칙:
- "채권", "단기채", "국공채", "회사채", "크레딧", "국채" -> 채권형
- "주식", "성장주", "배당주", "고배당" -> 주식형
- 명확하지 않으면 생략합니다.

위험등급 규칙:
한국 펀드 위험등급은 숫자가 작을수록 더 위험합니다.
1 = 매우 높은 위험
2 = 높은 위험
3~4 = 중간 수준
5 = 낮은 위험
6 = 매우 낮은 위험

자연어 매핑:
- "매우 안전", "위험을 최대한 낮게", "최저위험" -> 6~6
- "안전", "저위험", "보수적", "안정적" -> 5~6
- "중위험", "균형적" -> 3~4
- "공격적", "고위험" -> 1~2
- "매우 공격적" -> 1~1
- 단, "보수가 낮은", "수수료가 싼"의 '보수'는 위험성향이 아니라 비용입니다.
  절대 위험등급 5~6으로 해석하지 마세요.

채널 규칙:
- "온라인", "모바일", "비대면" -> online_only=true
- "온라인슈퍼" -> online_only=true, preferred_channel="online_super"
- "오프라인" -> preferred_channel="offline"

키워드 규칙:
- 사용자가 상품 특성을 말하면 펀드명 우선순위에 쓸 짧은 키워드만 넣습니다.
- 예: "단기채" -> ["단기"]
- "국공채" -> ["국공채"]
- "미국 고배당" -> ["미국", "고배당"]
- 계좌명, "추천", "안전", "온라인", "저렴" 같은 일반 조건은 keywords에 넣지 않습니다.

비용 규칙:
- 별도의 cost 필드는 만들지 않습니다.
- deterministic 추천기는 기본적으로 검증된 비용이 낮은 순으로 정렬합니다.
- 따라서 "수수료 싼", "보수 낮은" 요청은 다른 조건만 추출하면 됩니다.

top_k:
- "3개", "5개"처럼 명시되면 반영
- 없으면 5

지원 범위 밖:
- 예금, 보험, 개별주식, 가상자산 등 현재 펀드 DB 밖의 상품만 요구하면 unsupported.
- ETF만을 특정해서 요구하는 경우 현재 펀드 DB에서 ETF 여부를 확정할 수 없으므로 unsupported.
"""


# =========================================================
# 자연어 핵심조건 deterministic 보정
#
# LLM이 누락한 명시적 키워드를 보완한다.
# 추천/순위 자체는 여전히 recommend_products.py가 담당한다.
# =========================================================

QUERY_KEYWORD_RULES = [
    ("단기채", "단기"),
    ("초단기채", "단기"),
    ("초단기", "단기"),
    ("국공채", "국공채"),
    ("국채", "국채"),
    ("회사채", "회사채"),
    ("크레딧", "크레딧"),
    ("고배당", "고배당"),
    ("배당주", "배당"),
    ("배당", "배당"),
    ("미국", "미국"),
    ("국내", "국내"),
    ("코리아", "코리아"),
    ("글로벌", "글로벌"),
    ("인덱스", "인덱스"),
]


def apply_query_fallbacks(
    parsed: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    """
    HyperCLOVA X가 명시적 조건을 빠뜨렸을 때 최소한으로 보정한다.

    예:
      "IRP에서 안전한 단기채 위주로..."
      -> LLM이 keywords=[]로 반환해도 ["단기"]를 추가.

    추론이 아니라 사용자가 실제로 쓴 문자열만 기반으로 한다.
    """

    result = dict(parsed)

    query_compact = (
        query
        .replace(" ", "")
        .lower()
    )

    keywords = list(
        result.get(
            "keywords",
            [],
        )
        or []
    )

    for phrase, keyword in QUERY_KEYWORD_RULES:

        if (
            phrase.replace(" ", "").lower()
            in query_compact
            and keyword not in keywords
        ):
            keywords.append(
                keyword
            )

    result["keywords"] = keywords[:8]

    # -------------------------------------------------
    # 상품유형도 명시어가 있는데 LLM이 빠뜨린 경우만 보정
    # -------------------------------------------------

    if not result.get(
        "product_type"
    ):

        bond_terms = (
            "채권",
            "단기채",
            "초단기",
            "국공채",
            "국채",
            "회사채",
            "크레딧",
        )

        stock_terms = (
            "주식",
            "성장주",
            "배당주",
            "고배당",
        )

        if any(
            term in query
            for term in bond_terms
        ):
            result[
                "product_type"
            ] = "채권형"

        elif any(
            term in query
            for term in stock_terms
        ):
            result[
                "product_type"
            ] = "주식형"

    return result


# =========================================================
# HyperCLOVA X 호출
# =========================================================

def get_clova_client() -> OpenAI:
    api_key = os.getenv(
        "CLOVA_STUDIO_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY 환경변수가 없습니다.\n"
            "PowerShell 예시:\n"
            '$env:CLOVA_STUDIO_API_KEY="발급받은_API_KEY"'
        )

    return OpenAI(
        api_key=api_key,
        base_url=CLOVA_BASE_URL,
    )


def parse_user_query(
    query: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:

    query = query.strip()

    if not query:
        raise ValueError(
            "사용자 질문이 비어 있습니다."
        )

    client = get_clova_client()

    response = (
        client.chat.completions.create(
            model=model or CLOVA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            tools=[
                PARSE_TOOL
            ],
            tool_choice={
                "type": "function",
                "function": {
                    "name": (
                        "parse_recommendation_request"
                    )
                },
            },
            temperature=0,
            max_tokens=1024,
        )
    )

    message = (
        response
        .choices[0]
        .message
    )

    tool_calls = (
        message.tool_calls
        or []
    )

    if not tool_calls:
        raise RuntimeError(
            "HyperCLOVA X가 구조화 조건을 반환하지 않았습니다."
        )

    arguments = (
        tool_calls[0]
        .function
        .arguments
    )

    try:
        parsed = json.loads(
            arguments
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "HyperCLOVA X function arguments JSON 파싱 실패:\n"
            + arguments
        ) from exc

    parsed = sanitize_parsed_request(
        parsed
    )

    parsed = apply_query_fallbacks(
        parsed,
        query,
    )

    return parsed


# =========================================================
# LLM 출력 방어적 검증
# =========================================================

def sanitize_parsed_request(
    data: dict[str, Any],
) -> dict[str, Any]:

    result = dict(data)

    result.setdefault(
        "intent",
        "recommendation",
    )

    result.setdefault(
        "online_only",
        False,
    )

    result.setdefault(
        "top_k",
        5,
    )

    result.setdefault(
        "needs_clarification",
        False,
    )

    result.setdefault(
        "clarifying_question",
        "",
    )

    result.setdefault(
        "unsupported_reason",
        "",
    )

    result.setdefault(
        "interpretation",
        "",
    )

    result.setdefault(
        "keywords",
        [],
    )

    # ---------------------------------------------
    # enum 방어
    # ---------------------------------------------

    valid_accounts = {
        "IRP",
        "퇴직연금",
        "연금저축",
    }

    if (
        result.get("account_type")
        not in valid_accounts
    ):
        result.pop(
            "account_type",
            None,
        )

    valid_products = {
        "채권형",
        "주식형",
    }

    if (
        result.get("product_type")
        not in valid_products
    ):
        result.pop(
            "product_type",
            None,
        )

    valid_channels = {
        "online",
        "online_super",
        "offline",
        "default_option",
    }

    if (
        result.get("preferred_channel")
        not in valid_channels
    ):
        result.pop(
            "preferred_channel",
            None,
        )

    # ---------------------------------------------
    # 위험등급 방어
    # ---------------------------------------------

    for key in (
        "risk_grade_min",
        "risk_grade_max",
    ):

        if key not in result:
            continue

        try:
            value = int(
                result[key]
            )
        except (
            TypeError,
            ValueError,
        ):
            result.pop(
                key,
                None,
            )
            continue

        if not 1 <= value <= 6:
            result.pop(
                key,
                None,
            )
        else:
            result[key] = value

    risk_min = result.get(
        "risk_grade_min"
    )

    risk_max = result.get(
        "risk_grade_max"
    )

    if (
        risk_min is not None
        and risk_max is not None
        and risk_min > risk_max
    ):
        result[
            "risk_grade_min"
        ] = risk_max

        result[
            "risk_grade_max"
        ] = risk_min

    # ---------------------------------------------
    # 기타 필드 방어
    # ---------------------------------------------

    result[
        "online_only"
    ] = bool(
        result.get(
            "online_only",
            False,
        )
    )

    try:
        top_k = int(
            result.get(
                "top_k",
                5,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        top_k = 5

    result[
        "top_k"
    ] = max(
        1,
        min(
            10,
            top_k,
        ),
    )

    keywords = result.get(
        "keywords",
        [],
    )

    if not isinstance(
        keywords,
        list,
    ):
        keywords = [
            str(
                keywords
            )
        ]

    result[
        "keywords"
    ] = [
        str(item).strip()
        for item in keywords
        if str(item).strip()
    ][:8]

    # ---------------------------------------------
    # 실제 class 추천에는 계좌가 필요함.
    # LLM이 실수로 clarification을 빼먹어도 방어.
    # ---------------------------------------------

    if (
        result.get("intent")
        == "recommendation"
        and not result.get(
            "account_type"
        )
    ):
        result[
            "needs_clarification"
        ] = True

        if not result.get(
            "clarifying_question"
        ):
            result[
                "clarifying_question"
            ] = (
                "IRP, 퇴직연금, 연금저축 중 "
                "어떤 계좌에서 투자할 상품을 찾고 있나요?"
            )

    return result


# =========================================================
# 자연어 -> deterministic 추천
# =========================================================

def recommend_from_query(
    query: str,
    *,
    model: str | None = None,
    csv_path: str | None = None,
) -> dict[str, Any]:

    parsed = parse_user_query(
        query,
        model=model,
    )

    output = {
        "query": query,
        "parsed": parsed,
        "status": None,
        "recommendation_result": None,
    }

    if (
        parsed.get("intent")
        == "unsupported"
    ):
        output["status"] = (
            "unsupported"
        )
        return output

    if parsed.get(
        "needs_clarification"
    ):
        output["status"] = (
            "needs_clarification"
        )
        return output

    kwargs = {
        "account_type":
            parsed.get(
                "account_type"
            ),

        "product_type":
            parsed.get(
                "product_type"
            ),

        "risk_grade_min":
            parsed.get(
                "risk_grade_min"
            ),

        "risk_grade_max":
            parsed.get(
                "risk_grade_max"
            ),

        "online_only":
            parsed.get(
                "online_only",
                False,
            ),

        "preferred_channel":
            parsed.get(
                "preferred_channel"
            ),

        "keywords":
            parsed.get(
                "keywords"
            ),

        "top_k":
            parsed.get(
                "top_k",
                5,
            ),

        # 금융 상품 추천에서는
        # LLM이 이 값을 바꿀 수 없게 고정한다.
        "require_verified_cost":
            True,

        "strict_review":
            True,

        "csv_path":
            csv_path,
    }

    result = recommend_products(
        **kwargs
    )

    output["status"] = "ok"

    output[
        "recommendation_result"
    ] = result

    return output


# =========================================================
# CLI
# =========================================================

def print_parsed(
    parsed: dict[str, Any],
) -> None:

    print()
    print("=" * 80)
    print("HyperCLOVA X 해석 결과")
    print("=" * 80)

    print(
        json.dumps(
            parsed,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "HyperCLOVA X 자연어 조건 추출 + 명시어 보정 "
            "+ deterministic 연금상품 추천"
        )
    )

    parser.add_argument(
        "query",
        type=str,
        help=(
            '예: "IRP에서 안전한 단기채 위주로 '
            '온라인 상품 5개 추천해줘"'
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HyperCLOVA X 모델명. "
            "기본값은 CLOVA_STUDIO_MODEL "
            "환경변수 또는 HCX-005."
        ),
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "pension_classes.csv 직접 지정. "
            "생략하면 recommend_products.py가 자동 탐색."
        ),
    )

    args = parser.parse_args()

    result = recommend_from_query(
        args.query,
        model=args.model,
        csv_path=args.csv,
    )

    parsed = result[
        "parsed"
    ]

    print_parsed(
        parsed
    )

    status = result[
        "status"
    ]

    if status == "unsupported":

        print()
        print(
            "현재 추천 DB 범위 밖 요청입니다:"
        )

        print(
            parsed.get(
                "unsupported_reason",
                "",
            )
        )

        return

    if status == "needs_clarification":

        print()
        print("=" * 80)
        print("추가 확인 필요")
        print("=" * 80)

        print(
            parsed.get(
                "clarifying_question"
            )
        )

        return

    print_result(
        result[
            "recommendation_result"
        ]
    )


if __name__ == "__main__":
    main()
