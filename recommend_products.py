from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# =========================================================
# 기본 설정
# =========================================================

VALID_CHANNELS = {
    "online",
    "online_super",
    "offline",
    "default_option",
}

ONLINE_CHANNELS = {
    "online",
    "online_super",
}

BAD_REVIEW_PATTERNS = (
    "fee_component_sum_mismatch",
    "duplicate_fee_conflict",
)


# =========================================================
# 경로 자동 탐색
# =========================================================

def find_default_csv() -> Path:
    """
    현재 스크립트 위치 / PowerShell 실행 위치 기준으로
    _processed/pension_classes.csv 를 자동 탐색한다.
    """

    starts = [
        Path(__file__).resolve().parent,
        Path.cwd().resolve(),
    ]

    candidates: list[Path] = []

    for start in starts:
        current = start

        for _ in range(6):
            direct = current / "_processed" / "pension_classes.csv"

            if direct.exists():
                candidates.append(direct.resolve())

            try:
                for path in current.glob("**/_processed/pension_classes.csv"):
                    if path.exists():
                        candidates.append(path.resolve())
            except (PermissionError, OSError):
                pass

            if current.parent == current:
                break

            current = current.parent

        if candidates:
            break

    unique = {}
    for path in candidates:
        unique[str(path)] = path

    candidates = list(unique.values())

    if not candidates:
        raise FileNotFoundError(
            "pension_classes.csv를 자동으로 찾지 못했습니다.\n"
            "CSV 경로를 recommend_products(..., csv_path=...)로 직접 넘기거나 "
            "스크립트를 프로젝트 상위 폴더에 두세요."
        )

    # 같은 이름이 여러 개면 가장 최근 수정본 사용
    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return candidates[0]


# =========================================================
# 데이터 로드 / 정규화
# =========================================================

def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def load_product_classes(
    csv_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, Path]:

    path = (
        Path(csv_path).resolve()
        if csv_path is not None
        else find_default_csv()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"CSV를 찾지 못했습니다: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "source_folder_id",
        "fund_code",
        "fund_name",
        "risk_grade",
        "product_type",
        "class_code",
        "account_type",
        "personal_pension_eligible",
        "retirement_pension_eligible",
        "irp_explicit",
        "channel",
        "comparison_cost_pct",
        "comparison_cost_basis",
        "needs_review",
        "review_reason",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "pension_classes.csv에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    for column in (
        "personal_pension_eligible",
        "retirement_pension_eligible",
        "irp_explicit",
        "needs_review",
    ):
        df[column] = df[column].map(_to_bool)

    df["risk_grade"] = pd.to_numeric(
        df["risk_grade"],
        errors="coerce",
    )

    numeric_fee_columns = [
        "management_fee_pct",
        "sales_company_fee_pct",
        "trustee_fee_pct",
        "admin_fee_pct",
        "total_fee_pct",
        "other_expense_pct",
        "total_expense_ratio_pct",
        "synthetic_total_expense_ratio_pct",
        "comparison_cost_pct",
    ]

    for column in numeric_fee_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["review_reason"] = (
        df["review_reason"]
        .fillna("")
        .astype(str)
    )

    df["channel"] = (
        df["channel"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["fund_name"] = (
        df["fund_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["class_code"] = (
        df["class_code"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df, path


# =========================================================
# 사용자 조건 정규화
# =========================================================

def normalize_account_type(
    account_type: Optional[str],
) -> Optional[str]:

    if account_type is None:
        return None

    value = str(account_type).strip().lower()

    aliases = {
        "irp": "irp",
        "개인형irp": "irp",
        "개인형퇴직연금": "irp",
        "개인형 퇴직연금": "irp",

        "퇴직연금": "retirement_pension",
        "퇴직": "retirement_pension",
        "dc": "retirement_pension",
        "db": "retirement_pension",
        "retirement": "retirement_pension",
        "retirement_pension": "retirement_pension",

        "연금저축": "personal_pension",
        "개인연금": "personal_pension",
        "personal": "personal_pension",
        "personal_pension": "personal_pension",
    }

    if value not in aliases:
        raise ValueError(
            "account_type은 IRP / 퇴직연금 / 연금저축(개인연금) 중 하나여야 합니다."
        )

    return aliases[value]


def normalize_product_type(
    product_type: Optional[str],
) -> Optional[str]:

    if product_type is None:
        return None

    value = str(product_type).strip()

    aliases = {
        "채권": "채권형",
        "채권형": "채권형",
        "주식": "주식형",
        "주식형": "주식형",
        "주식혼합": "주식혼합-재간접형",
        "주식혼합형": "주식혼합-재간접형",
        "주식혼합-재간접형": "주식혼합-재간접형",
    }

    return aliases.get(
        value,
        value,
    )


def normalize_keywords(
    keywords: Optional[str | Iterable[str]],
) -> list[str]:

    if keywords is None:
        return []

    if isinstance(keywords, str):
        tokens = (
            keywords
            .replace(",", " ")
            .split()
        )
    else:
        tokens = [
            str(x)
            for x in keywords
        ]

    return [
        token.strip().lower()
        for token in tokens
        if token
        and token.strip()
    ]


# =========================================================
# 비용 데이터 검증
# =========================================================

def add_cost_guard_columns(
    df: pd.DataFrame,
    tolerance: float = 0.005,
) -> pd.DataFrame:
    """
    V3 파싱 결과를 그대로 신뢰하지 않고 비용 필드의
    논리적 일관성을 한 번 더 검사한다.

    기본 관계:
    1) 세부 보수 합 ≈ 총보수
    2) 총보수·비용 >= 총보수
    3) 합성 총보수·비용 >= 총보수·비용
       (자펀드/재간접 구조에서는 특히 중요)

    관계가 깨진 행은 추천 비용 정렬에서 제외한다.
    """

    result = df.copy()

    reasons = []
    safe_costs = []
    safe_bases = []

    component_columns = [
        "management_fee_pct",
        "sales_company_fee_pct",
        "trustee_fee_pct",
        "admin_fee_pct",
    ]

    for _, row in result.iterrows():

        row_reasons = []

        components = [
            row.get(column)
            for column in component_columns
            if pd.notna(
                row.get(column)
            )
        ]

        total_fee = row.get(
            "total_fee_pct"
        )

        ter = row.get(
            "total_expense_ratio_pct"
        )

        synthetic = row.get(
            "synthetic_total_expense_ratio_pct"
        )

        # 세부 보수가 3개 이상 잡힌 경우만 합계 검증
        if (
            len(components) >= 3
            and pd.notna(total_fee)
        ):

            component_sum = sum(
                float(x)
                for x in components
            )

            if abs(
                component_sum
                - float(total_fee)
            ) > tolerance:

                row_reasons.append(
                    "component_sum_mismatch"
                )

        # 총보수·비용은 총보수보다 작을 수 없음
        if (
            pd.notna(ter)
            and pd.notna(total_fee)
            and float(ter) + tolerance
            < float(total_fee)
        ):

            row_reasons.append(
                "ter_below_total_fee"
            )

        # 합성 총보수·비용은 일반 총보수·비용보다 작으면 이상
        if (
            pd.notna(synthetic)
            and pd.notna(ter)
            and float(synthetic) + tolerance
            < float(ter)
        ):

            row_reasons.append(
                "synthetic_below_ter"
            )

        # TER가 없더라도 synthetic이 총보수보다 작으면 이상
        if (
            pd.notna(synthetic)
            and pd.notna(total_fee)
            and float(synthetic) + tolerance
            < float(total_fee)
        ):

            row_reasons.append(
                "synthetic_below_total_fee"
            )

        original_review = str(
            row.get(
                "review_reason",
                "",
            )
        )

        if any(
            bad in original_review
            for bad in BAD_REVIEW_PATTERNS
        ):

            row_reasons.append(
                "parser_review_conflict"
            )

        row_reasons = list(
            dict.fromkeys(
                row_reasons
            )
        )

        # -------------------------------------------------
        # 안전한 비용 선택
        # -------------------------------------------------

        safe_cost = None
        safe_basis = None

        if not row_reasons:

            if pd.notna(synthetic):

                safe_cost = float(
                    synthetic
                )

                safe_basis = (
                    "synthetic_total_expense_ratio"
                )

            elif pd.notna(ter):

                safe_cost = float(
                    ter
                )

                safe_basis = (
                    "total_expense_ratio"
                )

            elif pd.notna(total_fee):

                safe_cost = float(
                    total_fee
                )

                safe_basis = (
                    "total_fee"
                )

        reasons.append(
            ",".join(
                row_reasons
            )
        )

        safe_costs.append(
            safe_cost
        )

        safe_bases.append(
            safe_basis
        )

    result[
        "_cost_guard_reason"
    ] = reasons

    result[
        "_safe_comparison_cost_pct"
    ] = safe_costs

    result[
        "_safe_comparison_cost_basis"
    ] = safe_bases

    result[
        "_cost_guard_ok"
    ] = (
        result[
            "_safe_comparison_cost_pct"
        ].notna()
    )

    return result


# =========================================================
# 추천 로직
# =========================================================

def recommend_products(
    *,
    account_type: Optional[str] = None,
    product_type: Optional[str] = None,
    risk_grades: Optional[Iterable[int]] = None,
    risk_grade_min: Optional[int] = None,
    risk_grade_max: Optional[int] = None,
    online_only: bool = False,
    preferred_channel: Optional[str] = None,
    keywords: Optional[str | Iterable[str]] = None,
    top_k: int = 5,
    require_verified_cost: bool = True,
    strict_review: bool = True,
    csv_path: Optional[str | Path] = None,
) -> dict:
    """
    연금 상품 추천 후보 생성기.

    중요:
    - LLM 점수로 추천하지 않는다.
    - 계좌/상품유형/위험등급/채널은 deterministic hard filter.
    - 그 후 펀드별 가장 적합한 class 하나를 선택.
    - 비용 논리 검증을 통과한 후보끼리 안전한 비교비용 오름차순 정렬.
    - IRP 요청 시 irp_explicit=True만 사용한다.

    위험등급:
    1 = 매우 높은 위험
    ...
    6 = 매우 낮은 위험

    따라서 risk_grade_min=5, risk_grade_max=6 이면
    낮은 위험~매우 낮은 위험 범위다.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k는 1 이상이어야 합니다."
        )

    normalized_account = normalize_account_type(
        account_type
    )

    normalized_product = normalize_product_type(
        product_type
    )

    keyword_list = normalize_keywords(
        keywords
    )

    if preferred_channel is not None:
        preferred_channel = str(
            preferred_channel
        ).strip()

        if preferred_channel not in VALID_CHANNELS:
            raise ValueError(
                "preferred_channel은 "
                "online / online_super / offline / default_option 중 하나여야 합니다."
            )

    if risk_grade_min is not None:
        risk_grade_min = int(
            risk_grade_min
        )

    if risk_grade_max is not None:
        risk_grade_max = int(
            risk_grade_max
        )

    if (
        risk_grade_min is not None
        and not 1 <= risk_grade_min <= 6
    ):
        raise ValueError(
            "risk_grade_min은 1~6이어야 합니다."
        )

    if (
        risk_grade_max is not None
        and not 1 <= risk_grade_max <= 6
    ):
        raise ValueError(
            "risk_grade_max은 1~6이어야 합니다."
        )

    if (
        risk_grade_min is not None
        and risk_grade_max is not None
        and risk_grade_min > risk_grade_max
    ):
        raise ValueError(
            "risk_grade_min은 risk_grade_max보다 클 수 없습니다."
        )

    grades = None

    if risk_grades is not None:
        grades = sorted(
            {
                int(x)
                for x in risk_grades
            }
        )

        if any(
            grade < 1 or grade > 6
            for grade in grades
        ):
            raise ValueError(
                "risk_grades는 1~6 범위여야 합니다."
            )

    df, used_csv = load_product_classes(
        csv_path
    )

    df = add_cost_guard_columns(
        df
    )

    diagnostics = {
        "input_class_rows": int(
            len(df)
        ),
        "cost_guard_rejected_rows": int(
            (
                (
                    df["comparison_cost_pct"].notna()
                )
                & (
                    ~df["_cost_guard_ok"]
                )
            ).sum()
        ),
    }

    work = df.copy()

    # -----------------------------------------------------
    # 1. 계좌 eligibility
    # -----------------------------------------------------

    if normalized_account == "irp":
        work = work[
            work[
                "retirement_pension_eligible"
            ]
            & work[
                "irp_explicit"
            ]
        ]

    elif normalized_account == "retirement_pension":
        work = work[
            work[
                "retirement_pension_eligible"
            ]
        ]

    elif normalized_account == "personal_pension":
        work = work[
            work[
                "personal_pension_eligible"
            ]
        ]

    diagnostics[
        "after_account_filter"
    ] = int(
        len(work)
    )

    # -----------------------------------------------------
    # 2. 상품 유형
    # -----------------------------------------------------

    if normalized_product is not None:
        work = work[
            work[
                "product_type"
            ].astype(str)
            == normalized_product
        ]

    diagnostics[
        "after_product_type_filter"
    ] = int(
        len(work)
    )

    # -----------------------------------------------------
    # 3. 위험등급
    # -----------------------------------------------------

    if grades is not None:
        work = work[
            work[
                "risk_grade"
            ].isin(
                grades
            )
        ]

    if risk_grade_min is not None:
        work = work[
            work[
                "risk_grade"
            ]
            >= risk_grade_min
        ]

    if risk_grade_max is not None:
        work = work[
            work[
                "risk_grade"
            ]
            <= risk_grade_max
        ]

    diagnostics[
        "after_risk_filter"
    ] = int(
        len(work)
    )

    # -----------------------------------------------------
    # 4. 채널
    # -----------------------------------------------------

    if online_only:
        work = work[
            work[
                "channel"
            ].isin(
                ONLINE_CHANNELS
            )
        ]

    diagnostics[
        "after_channel_filter"
    ] = int(
        len(work)
    )

    # -----------------------------------------------------
    # 5. 비용 검증
    # -----------------------------------------------------

    if require_verified_cost:
        work = work[
            work[
                "_safe_comparison_cost_pct"
            ].notna()
        ]

    diagnostics[
        "after_cost_filter"
    ] = int(
        len(work)
    )

    # -----------------------------------------------------
    # 6. review 품질
    # -----------------------------------------------------

    if strict_review:
        bad_mask = work[
            "review_reason"
        ].str.contains(
            "|".join(
                BAD_REVIEW_PATTERNS
            ),
            regex=True,
            na=False,
        )

        # annual_fee_not_found는 comparison_cost_pct가 존재하면
        # 이미 위 단계에서 걸러졌으므로 여기서는
        # 숫자 충돌 / 중복 충돌만 제거한다.
        work = work[
            ~bad_mask
        ]

    diagnostics[
        "after_review_filter"
    ] = int(
        len(work)
    )

    if work.empty:
        return {
            "csv_path": str(
                used_csv
            ),
            "conditions": {
                "account_type":
                    normalized_account,
                "product_type":
                    normalized_product,
                "risk_grades":
                    grades,
                "risk_grade_min":
                    risk_grade_min,
                "risk_grade_max":
                    risk_grade_max,
                "online_only":
                    online_only,
                "preferred_channel":
                    preferred_channel,
                "keywords":
                    keyword_list,
                "require_verified_cost":
                    require_verified_cost,
                "strict_review":
                    strict_review,
                "top_k":
                    top_k,
            },
            "diagnostics":
                diagnostics,
            "recommendations":
                [],
        }

    # -----------------------------------------------------
    # 7. 펀드 내부에서 가장 적합한 class 하나 선택
    # -----------------------------------------------------

    if preferred_channel is not None:
        work[
            "_channel_rank"
        ] = (
            work[
                "channel"
            ]
            != preferred_channel
        ).astype(int)

    elif online_only:
        # 온라인슈퍼를 무조건 더 좋다고 간주하지 않고
        # 둘 다 같은 우선순위.
        work[
            "_channel_rank"
        ] = 0

    else:
        work[
            "_channel_rank"
        ] = 0

    work[
        "_review_rank"
    ] = work[
        "needs_review"
    ].astype(int)

    # 같은 fund 내에서는:
    # 선호채널 -> clean row -> 비용 낮은 class
    work = work.sort_values(
        by=[
            "source_folder_id",
            "_channel_rank",
            "_review_rank",
            "_safe_comparison_cost_pct",
            "class_code",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
        na_position="last",
    )

    best_class_per_fund = (
        work
        .drop_duplicates(
            subset=[
                "source_folder_id"
            ],
            keep="first",
        )
        .copy()
    )

    diagnostics[
        "unique_funds_after_filters"
    ] = int(
        len(
            best_class_per_fund
        )
    )

    # -----------------------------------------------------
    # 8. 키워드 적합도
    #
    # 가중 점수는 쓰지 않고 단순 match 개수만
    # lexicographic 우선순위로 사용.
    # -----------------------------------------------------

    if keyword_list:

        def keyword_match_count(row) -> int:
            haystack = " ".join(
                [
                    str(
                        row.get(
                            "fund_name",
                            "",
                        )
                    ),
                    str(
                        row.get(
                            "class_name_raw",
                            "",
                        )
                    ),
                    str(
                        row.get(
                            "product_type",
                            "",
                        )
                    ),
                ]
            ).lower()

            return sum(
                1
                for token in keyword_list
                if token in haystack
            )

        best_class_per_fund[
            "_keyword_matches"
        ] = best_class_per_fund.apply(
            keyword_match_count,
            axis=1,
        )

    else:
        best_class_per_fund[
            "_keyword_matches"
        ] = 0

    # -----------------------------------------------------
    # 9. 최종 deterministic ranking
    #
    # keyword fit -> 비용 -> 더 낮은 위험(등급 숫자 큼)
    # -> 이름
    # -----------------------------------------------------

    ranked = (
        best_class_per_fund
        .sort_values(
            by=[
                "_keyword_matches",
                "_safe_comparison_cost_pct",
                "risk_grade",
                "fund_name",
            ],
            ascending=[
                False,
                True,
                False,
                True,
            ],
            na_position="last",
        )
        .head(
            top_k
        )
    )

    recommendations = []

    for rank, (_, row) in enumerate(
        ranked.iterrows(),
        start=1,
    ):

        reasons = []

        if normalized_account == "irp":
            reasons.append(
                "투자설명서에서 IRP/개인퇴직계좌 관련 근거 확인"
            )
        elif normalized_account == "retirement_pension":
            reasons.append(
                "퇴직연금 가입 가능 클래스"
            )
        elif normalized_account == "personal_pension":
            reasons.append(
                "연금저축/개인연금 가입 가능 클래스"
            )

        if normalized_product:
            reasons.append(
                normalized_product
            )

        if pd.notna(
            row[
                "risk_grade"
            ]
        ):
            reasons.append(
                f"위험등급 {int(row['risk_grade'])}등급"
            )

        if row.get(
            "channel"
        ):
            reasons.append(
                f"채널 {row['channel']}"
            )

        if pd.notna(
            row[
                "_safe_comparison_cost_pct"
            ]
        ):
            reasons.append(
                f"검증비용 {row['_safe_comparison_cost_pct']:.4f}%"
            )

        recommendations.append(
            {
                "rank":
                    rank,

                "source_folder_id":
                    row.get(
                        "source_folder_id"
                    ),

                "fund_code":
                    row.get(
                        "fund_code"
                    ),

                "fund_name":
                    row.get(
                        "fund_name"
                    ),

                "asset_manager":
                    row.get(
                        "asset_manager"
                    ),

                "product_type":
                    row.get(
                        "product_type"
                    ),

                "risk_grade":
                    (
                        int(
                            row[
                                "risk_grade"
                            ]
                        )
                        if pd.notna(
                            row[
                                "risk_grade"
                            ]
                        )
                        else None
                    ),

                "class_code":
                    row.get(
                        "class_code"
                    ),

                "class_name_raw":
                    row.get(
                        "class_name_raw"
                    ),

                "channel":
                    row.get(
                        "channel"
                    ),

                "account_type":
                    row.get(
                        "account_type"
                    ),

                "personal_pension_eligible":
                    bool(
                        row.get(
                            "personal_pension_eligible"
                        )
                    ),

                "retirement_pension_eligible":
                    bool(
                        row.get(
                            "retirement_pension_eligible"
                        )
                    ),

                "irp_explicit":
                    bool(
                        row.get(
                            "irp_explicit"
                        )
                    ),

                "comparison_cost_pct":
                    (
                        float(
                            row[
                                "_safe_comparison_cost_pct"
                            ]
                        )
                        if pd.notna(
                            row[
                                "_safe_comparison_cost_pct"
                            ]
                        )
                        else None
                    ),

                "comparison_cost_basis":
                    row.get(
                        "_safe_comparison_cost_basis"
                    ),

                "raw_comparison_cost_pct":
                    (
                        float(
                            row[
                                "comparison_cost_pct"
                            ]
                        )
                        if pd.notna(
                            row[
                                "comparison_cost_pct"
                            ]
                        )
                        else None
                    ),

                "raw_comparison_cost_basis":
                    row.get(
                        "comparison_cost_basis"
                    ),

                "cost_guard_reason":
                    row.get(
                        "_cost_guard_reason"
                    ),

                "effective_date":
                    row.get(
                        "effective_date"
                    ),

                "fee_source_page":
                    (
                        int(
                            row[
                                "fee_source_page"
                            ]
                        )
                        if pd.notna(
                            row.get(
                                "fee_source_page"
                            )
                        )
                        else None
                    ),

                "eligibility_source_page":
                    (
                        int(
                            row[
                                "eligibility_source_page"
                            ]
                        )
                        if pd.notna(
                            row.get(
                                "eligibility_source_page"
                            )
                        )
                        else None
                    ),

                "source_pdf":
                    row.get(
                        "source_pdf"
                    ),

                "confidence":
                    row.get(
                        "confidence"
                    ),

                "selection_reason":
                    reasons,
            }
        )

    return {
        "csv_path": str(
            used_csv
        ),
        "conditions": {
            "account_type":
                normalized_account,
            "product_type":
                normalized_product,
            "risk_grades":
                grades,
            "risk_grade_min":
                risk_grade_min,
            "risk_grade_max":
                risk_grade_max,
            "online_only":
                online_only,
            "preferred_channel":
                preferred_channel,
            "keywords":
                keyword_list,
            "require_verified_cost":
                require_verified_cost,
            "strict_review":
                strict_review,
            "top_k":
                top_k,
        },
        "diagnostics":
            diagnostics,
        "recommendations":
            recommendations,
    }


# =========================================================
# CLI 출력
# =========================================================

def print_result(
    result: dict,
) -> None:

    print()
    print("=" * 80)
    print("추천 조건")
    print("=" * 80)

    for key, value in result[
        "conditions"
    ].items():
        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 80)
    print("필터 진단")
    print("=" * 80)

    for key, value in result[
        "diagnostics"
    ].items():
        print(
            f"{key}: {value}"
        )

    recommendations = result[
        "recommendations"
    ]

    print()
    print("=" * 80)
    print(
        f"추천 결과: {len(recommendations)}개"
    )
    print("=" * 80)

    if not recommendations:
        print(
            "조건을 만족하는 검증된 상품을 찾지 못했습니다."
        )
        return

    for item in recommendations:

        print()
        print(
            f"[{item['rank']}] "
            f"{item['fund_name']}"
        )

        print(
            f"  운용사: {item['asset_manager']}"
        )

        print(
            f"  상품유형: {item['product_type']}"
        )

        print(
            f"  위험등급: {item['risk_grade']}"
        )

        print(
            f"  class: {item['class_code']}"
        )

        print(
            f"  채널: {item['channel']}"
        )

        print(
            f"  IRP 명시: {item['irp_explicit']}"
        )

        print(
            f"  비교비용: "
            f"{item['comparison_cost_pct']}% "
            f"({item['comparison_cost_basis']})"
        )

        print(
            "  선정근거: "
            + " / ".join(
                item[
                    "selection_reason"
                ]
            )
        )

        print(
            f"  비용 근거 페이지: "
            f"{item['fee_source_page']}"
        )

        print(
            f"  가입자격 근거 페이지: "
            f"{item['eligibility_source_page']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "연금/IRP 상품 deterministic 추천 후보 생성기"
        )
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "pension_classes.csv 경로. "
            "생략하면 자동 탐색."
        ),
    )

    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help=(
            "IRP / 퇴직연금 / 연금저축"
        ),
    )

    parser.add_argument(
        "--product-type",
        type=str,
        default=None,
        help=(
            "채권형 / 주식형 / 주식혼합-재간접형"
        ),
    )

    parser.add_argument(
        "--risk-grades",
        type=int,
        nargs="*",
        default=None,
        help=(
            "허용 위험등급 목록. 예: --risk-grades 5 6"
        ),
    )

    parser.add_argument(
        "--risk-min",
        type=int,
        default=None,
        help=(
            "최소 위험등급 숫자. "
            "예: 5이면 5~6등급"
        ),
    )

    parser.add_argument(
        "--risk-max",
        type=int,
        default=None,
        help=(
            "최대 위험등급 숫자."
        ),
    )

    parser.add_argument(
        "--online-only",
        action="store_true",
        help=(
            "online / online_super class만 허용"
        ),
    )

    parser.add_argument(
        "--preferred-channel",
        type=str,
        default=None,
        help=(
            "online / online_super / offline / default_option"
        ),
    )

    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help=(
            "펀드명 keyword 우선순위. "
            '예: --keywords "단기 국공채"'
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--allow-unverified-cost",
        action="store_true",
        help=(
            "비용이 없는 class도 후보에 허용"
        ),
    )

    parser.add_argument(
        "--allow-review",
        action="store_true",
        help=(
            "fee mismatch/conflict가 있는 class도 허용"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "결과를 JSON으로 출력"
        ),
    )

    return parser


def main() -> None:

    parser = build_arg_parser()

    args = parser.parse_args()

    result = recommend_products(
        account_type=
            args.account,
        product_type=
            args.product_type,
        risk_grades=
            args.risk_grades,
        risk_grade_min=
            args.risk_min,
        risk_grade_max=
            args.risk_max,
        online_only=
            args.online_only,
        preferred_channel=
            args.preferred_channel,
        keywords=
            args.keywords,
        top_k=
            args.top_k,
        require_verified_cost=
            not args.allow_unverified_cost,
        strict_review=
            not args.allow_review,
        csv_path=
            args.csv,
    )

    if args.json:

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    else:

        print_result(
            result
        )


if __name__ == "__main__":
    main()
