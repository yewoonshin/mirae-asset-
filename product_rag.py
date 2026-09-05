from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


# =========================================================
# 경로
# =========================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_RAG_FILE = ROOT / "_processed" / "rag_chunks.jsonl"


# 실제 rag_chunks.jsonl section 값 기준
DEFAULT_SECTIONS = (
    "strategy",
    "risk",
    "objective",
    "assets",
    "purchase_redemption",
)


# 최종 설명에 필요한 evidence를 section별로 강제 확보하기 위한 설정.
# risk_grade / 비용 / class / 계좌 가입 가능 여부 / channel은
# pension_classes.csv 및 recommend_products.py 결과를 truth source로 사용한다.
CATEGORY_SPECS = {
    "objective": {
        "sections": ("objective",),
        "query_hint": "투자목적 안정적 수익 투자수익 추구 원금보장",
        "k": 1,
    },
    "assets": {
        "sections": ("assets",),
        "query_hint": (
            "투자대상 투자비율 채권 주식 국공채 회사채 "
            "모투자신탁 유동성자산"
        ),
        "k": 1,
    },
    "strategy": {
        "sections": ("strategy",),
        "query_hint": (
            "투자전략 운용전략 투자방침 분산투자 "
            "듀레이션 금리 신용 채권 이자수익"
        ),
        "k": 1,
    },
    "risk": {
        "sections": ("risk",),
        "query_hint": (
            "투자위험 주요위험 원금손실 금리변동 신용위험 "
            "시장위험 유동성위험 채권가격"
        ),
        "k": 2,
    },
    "purchase_redemption": {
        "sections": ("purchase_redemption",),
        "query_hint": (
            "환매 환매청구 환매대금 환매연기 "
            "매입 기준가격"
        ),
        "k": 1,
    },

    # 기본 bundle에는 넣지 않지만 필요할 때 --include 로 추가 가능
    "fees": {
        "sections": ("fees",),
        "query_hint": "보수 수수료 비용 총보수 합성총보수",
        "k": 1,
    },
    "tax": {
        "sections": ("tax",),
        "query_hint": "과세 세금 연금소득세 중도인출 연금수령",
        "k": 1,
    },
    "valuation": {
        "sections": ("valuation",),
        "query_hint": "기준가격 평가 산정 순자산",
        "k": 1,
    },
}


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
SPACE_RE = re.compile(r"\s+")


# =========================================================
# 문자열 / 검색 유틸
# =========================================================

def _norm_text(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]+", "", _norm_text(text))


def _terms(text: str) -> list[str]:
    """
    단순 whitespace token만 쓰면
    '단기채' / '단기 채권' / '단기채증권'처럼 한국어 복합어에 약하다.

    그래서 exact/sub-string term score와
    character n-gram coverage를 함께 사용한다.
    """
    out: list[str] = []
    seen: set[str] = set()

    for token in TOKEN_RE.findall(_norm_text(text)):
        token = token.strip()

        if len(token) < 2:
            continue

        if token not in seen:
            seen.add(token)
            out.append(token)

    return out


def _char_ngrams(text: str, n: int) -> set[str]:
    s = _compact(text)

    if len(s) < n:
        return {s} if s else set()

    return {
        s[i : i + n]
        for i in range(len(s) - n + 1)
    }


def _similarity(query: str, text: str) -> float:
    """
    외부 라이브러리/API 없이 동작하는 deterministic lexical scorer.

    구성:
    - query term이 document 안에 실제로 포함되는지
    - 한국어 character bigram / trigram coverage

    전체 100개 fund를 검색하지 않고
    source_folder_id로 먼저 metadata filtering한 뒤
    약 17~37개 chunk 안에서만 계산하므로 충분히 가볍다.
    """

    q_terms = _terms(query)

    if not q_terms:
        return 0.0

    d_compact = _compact(text)

    if not d_compact:
        return 0.0

    # -----------------------------------------------------
    # 1. query term coverage
    # 긴 단어가 일치했을 때 조금 더 높은 가중치
    # -----------------------------------------------------

    matched_weight = 0.0
    total_weight = 0.0

    for term in q_terms:
        t = _compact(term)

        if not t:
            continue

        weight = min(max(len(t), 2), 8)

        total_weight += weight

        if t in d_compact:
            matched_weight += weight

    term_coverage = (
        matched_weight / total_weight
        if total_weight
        else 0.0
    )

    # -----------------------------------------------------
    # 2. Korean-friendly char n-gram coverage
    # -----------------------------------------------------

    q2 = _char_ngrams(query, 2)
    q3 = _char_ngrams(query, 3)

    d2 = _char_ngrams(text, 2)
    d3 = _char_ngrams(text, 3)

    bi_cov = (
        len(q2 & d2) / len(q2)
        if q2
        else 0.0
    )

    tri_cov = (
        len(q3 & d3) / len(q3)
        if q3
        else 0.0
    )

    return (
        0.65 * term_coverage
        + 0.20 * bi_cov
        + 0.15 * tri_cov
    )


def _page_label(
    start_page: Any,
    end_page: Any,
) -> str:

    if start_page is None and end_page is None:
        return ""

    if end_page is None or start_page == end_page:
        return f"p.{start_page}"

    return f"pp.{start_page}-{end_page}"


# =========================================================
# Product RAG
# =========================================================

class ProductRAG:

    def __init__(
        self,
        rag_file: str | Path = DEFAULT_RAG_FILE,
    ):
        self.rag_file = Path(rag_file)

        if not self.rag_file.exists():
            raise FileNotFoundError(
                "rag_chunks.jsonl을 찾을 수 없습니다.\n"
                f"path={self.rag_file}\n"
                "프로젝트 루트에서 실행하거나 "
                "--rag-file 경로를 직접 지정하세요."
            )

        self.rows: list[dict[str, Any]] = []

        self.by_folder: dict[
            str,
            list[dict[str, Any]]
        ] = defaultdict(list)

        self.folders_by_fund_code: dict[
            str,
            set[str]
        ] = defaultdict(set)

        self._load()


    # -----------------------------------------------------
    # JSONL load / index
    # -----------------------------------------------------

    def _load(self) -> None:

        required = {
            "chunk_id",
            "source_folder_id",
            "fund_code",
            "fund_name",
            "section",
            "start_page",
            "end_page",
            "source_pdf",
            "text",
        }

        with self.rag_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_no, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)

                except json.JSONDecodeError as e:

                    raise ValueError(
                        "JSONL 파싱 실패\n"
                        f"file={self.rag_file}\n"
                        f"line={line_no}\n"
                        f"error={e}"
                    ) from e


                missing = required - set(row)

                if missing:

                    raise ValueError(
                        "rag chunk 필수 필드 누락\n"
                        f"line={line_no}\n"
                        f"missing={sorted(missing)}"
                    )


                folder_id = str(
                    row["source_folder_id"]
                )

                fund_code = str(
                    row["fund_code"]
                )


                self.rows.append(row)

                self.by_folder[
                    folder_id
                ].append(row)

                self.folders_by_fund_code[
                    fund_code
                ].add(folder_id)


    # -----------------------------------------------------
    # fund 식별
    # -----------------------------------------------------

    def resolve_folder_id(
        self,
        *,
        source_folder_id: str | None = None,
        fund_code: str | None = None,
    ) -> str:

        # 가장 안전한 primary key
        if source_folder_id:

            if source_folder_id not in self.by_folder:

                raise KeyError(
                    "RAG에 없는 source_folder_id입니다: "
                    f"{source_folder_id}"
                )

            return source_folder_id


        if not fund_code:

            raise ValueError(
                "source_folder_id 또는 "
                "fund_code 중 하나는 필요합니다."
            )


        folders = sorted(
            self.folders_by_fund_code.get(
                str(fund_code),
                set(),
            )
        )


        if not folders:

            raise KeyError(
                "RAG에 없는 fund_code입니다: "
                f"{fund_code}"
            )


        # 실제 데이터에서 fund_code가 중복되는 경우가 존재한다.
        # 이 경우 임의로 하나를 고르지 않는다.
        if len(folders) > 1:

            raise ValueError(
                f"fund_code={fund_code}가 "
                "여러 상품에 매칭됩니다.\n"
                f"source_folder_ids={folders}\n"
                "recommend_products.py 결과의 "
                "source_folder_id를 사용하세요."
            )


        return folders[0]


    # -----------------------------------------------------
    # fund metadata
    # -----------------------------------------------------

    def fund_metadata(
        self,
        source_folder_id: str,
    ) -> dict[str, Any]:

        rows = self.by_folder[
            source_folder_id
        ]

        first = rows[0]

        return {
            "source_folder_id":
                source_folder_id,

            "fund_code":
                first.get("fund_code"),

            "fund_name":
                first.get("fund_name"),

            "asset_manager":
                first.get("asset_manager"),

            "risk_grade":
                first.get("risk_grade"),

            "product_type":
                first.get("product_type"),

            "effective_date":
                first.get("effective_date"),

            "source_pdf":
                first.get("source_pdf"),

            "chunk_count":
                len(rows),
        }


    # -----------------------------------------------------
    # 단일 fund 내부 검색
    # -----------------------------------------------------

    def search(
        self,
        *,
        query: str,
        source_folder_id: str | None = None,
        fund_code: str | None = None,
        sections: Iterable[str] | None = None,
        top_k: int = 5,
        query_hint: str = "",
    ) -> list[dict[str, Any]]:

        folder_id = self.resolve_folder_id(
            source_folder_id=source_folder_id,
            fund_code=fund_code,
        )


        allowed = set(
            sections or DEFAULT_SECTIONS
        )


        # 핵심:
        # 전체 2,493 chunks 검색 X
        # 해당 추천 fund의 chunks만 검색
        candidates = [
            row
            for row in self.by_folder[
                folder_id
            ]
            if row.get("section") in allowed
        ]


        combined_query = (
            f"{query} {query_hint}"
        ).strip()


        scored: list[
            tuple[
                float,
                dict[str, Any],
            ]
        ] = []


        for row in candidates:

            score = _similarity(
                combined_query,
                row["text"],
            )

            scored.append(
                (
                    score,
                    row,
                )
            )


        # deterministic tie-break
        scored.sort(
            key=lambda x: (
                -x[0],
                str(
                    x[1].get(
                        "section",
                        "",
                    )
                ),
                int(
                    x[1].get(
                        "chunk_index",
                        0,
                    )
                ),
                str(
                    x[1].get(
                        "chunk_id",
                        "",
                    )
                ),
            )
        )


        results: list[
            dict[str, Any]
        ] = []


        for score, row in scored[
            : max(top_k, 0)
        ]:

            results.append(
                {
                    "chunk_id":
                        row["chunk_id"],

                    "source_folder_id":
                        row["source_folder_id"],

                    "fund_code":
                        row["fund_code"],

                    "fund_name":
                        row["fund_name"],

                    "section":
                        row["section"],

                    "chunk_index":
                        row.get("chunk_index"),

                    "start_page":
                        row.get("start_page"),

                    "end_page":
                        row.get("end_page"),

                    "page_label":
                        _page_label(
                            row.get(
                                "start_page"
                            ),
                            row.get(
                                "end_page"
                            ),
                        ),

                    "effective_date":
                        row.get(
                            "effective_date"
                        ),

                    "source_pdf":
                        row.get(
                            "source_pdf"
                        ),

                    "score":
                        round(
                            score,
                            6,
                        ),

                    "text":
                        row["text"],
                }
            )


        return results


    # -----------------------------------------------------
    # 최종 설명용 evidence bundle
    # -----------------------------------------------------

    def retrieve_product_evidence(
        self,
        *,
        user_query: str,
        source_folder_id: str | None = None,
        fund_code: str | None = None,
        include_optional: Iterable[str] | None = None,
    ) -> dict[str, Any]:

        """
        최종 HyperCLOVA X 설명에 넣을 evidence bundle.

        기본적으로:
        - objective 1
        - assets 1
        - strategy 1
        - risk 2
        - purchase_redemption 1

        을 반환한다.

        risk_grade / 비용 / class_code /
        account eligibility / channel은
        여기서 재추출하지 않는다.
        """

        folder_id = self.resolve_folder_id(
            source_folder_id=source_folder_id,
            fund_code=fund_code,
        )


        categories = [
            "objective",
            "assets",
            "strategy",
            "risk",
            "purchase_redemption",
        ]


        for name in (
            include_optional or ()
        ):

            if name not in CATEGORY_SPECS:

                raise ValueError(
                    "지원하지 않는 "
                    "optional evidence category: "
                    f"{name}\n"
                    "가능값="
                    f"{sorted(CATEGORY_SPECS)}"
                )

            if name not in categories:

                categories.append(
                    name
                )


        evidence: list[
            dict[str, Any]
        ] = []

        seen_chunk_ids: set[str] = set()


        for category in categories:

            spec = CATEGORY_SPECS[
                category
            ]


            hits = self.search(
                query=user_query,
                source_folder_id=folder_id,
                sections=spec[
                    "sections"
                ],
                top_k=int(
                    spec["k"]
                ) + 2,
                query_hint=spec[
                    "query_hint"
                ],
            )


            taken = 0


            for hit in hits:

                if (
                    hit["chunk_id"]
                    in seen_chunk_ids
                ):
                    continue


                item = dict(hit)

                item[
                    "evidence_type"
                ] = category


                evidence.append(
                    item
                )

                seen_chunk_ids.add(
                    hit["chunk_id"]
                )

                taken += 1


                if taken >= int(
                    spec["k"]
                ):
                    break


        return {
            "product":
                self.fund_metadata(
                    folder_id
                ),

            "evidence":
                evidence,
        }


    # -----------------------------------------------------
    # recommend_products.py 결과 dict를 바로 넣기 위한 helper
    # -----------------------------------------------------

    def retrieve_for_recommendation(
        self,
        *,
        user_query: str,
        recommendation: dict[str, Any],
        include_optional: Iterable[str] | None = None,
    ) -> dict[str, Any]:

        source_folder_id = (
            recommendation.get(
                "source_folder_id"
            )
        )

        fund_code = (
            recommendation.get(
                "fund_code"
            )
        )


        if not source_folder_id and not fund_code:

            raise ValueError(
                "recommendation dict에 "
                "source_folder_id 또는 "
                "fund_code가 없습니다."
            )


        return self.retrieve_product_evidence(
            user_query=user_query,
            source_folder_id=source_folder_id,
            fund_code=(
                None
                if source_folder_id
                else fund_code
            ),
            include_optional=include_optional,
        )


# =========================================================
# CLI 출력
# =========================================================

def _print_human(
    bundle: dict[str, Any],
    max_chars: int,
) -> None:

    product = bundle[
        "product"
    ]


    print("=" * 80)

    print(
        product[
            "fund_name"
        ]
    )

    print(
        "source_folder_id="
        f"{product['source_folder_id']}"
        " | fund_code="
        f"{product['fund_code']}"
        " | chunks="
        f"{product['chunk_count']}"
    )

    print("=" * 80)


    for idx, ev in enumerate(
        bundle["evidence"],
        start=1,
    ):

        text = _norm_text(
            ev["text"]
        )


        if (
            max_chars > 0
            and len(text) > max_chars
        ):

            text = (
                text[:max_chars]
                .rstrip()
                + " ..."
            )


        print(
            "\n"
            f"[{idx}] "
            f"{ev['evidence_type']}"
            " / "
            f"{ev['section']}"
            " / "
            f"{ev['page_label']}"
            " / score="
            f"{ev['score']}"
        )

        print(text)


# =========================================================
# main
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "추천된 단일 fund 내부에서 "
            "투자설명서 RAG evidence를 검색합니다."
        )
    )


    parser.add_argument(
        "--rag-file",
        default=str(
            DEFAULT_RAG_FILE
        ),
        help=(
            "rag_chunks.jsonl 경로. "
            "기본값: "
            "_processed/rag_chunks.jsonl"
        ),
    )


    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--source-folder-id"
    )

    group.add_argument(
        "--fund-code"
    )


    parser.add_argument(
        "--query",
        required=True,
        help="원래 사용자 요청 문장",
    )


    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        choices=sorted(
            CATEGORY_SPECS
        ),
        help=(
            "기본 evidence 외 "
            "추가 section. "
            "예: --include fees tax"
        ),
    )


    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "사람용 출력 대신 "
            "JSON으로 출력"
        ),
    )


    parser.add_argument(
        "--max-chars",
        type=int,
        default=900,
        help=(
            "사람용 출력에서 "
            "evidence당 최대 표시 글자 수. "
            "0이면 전체 표시"
        ),
    )


    args = parser.parse_args()


    rag = ProductRAG(
        args.rag_file
    )


    bundle = (
        rag.retrieve_product_evidence(
            user_query=args.query,
            source_folder_id=(
                args.source_folder_id
            ),
            fund_code=(
                args.fund_code
            ),
            include_optional=(
                args.include
            ),
        )
    )


    if args.json:

        print(
            json.dumps(
                bundle,
                ensure_ascii=False,
                indent=2,
            )
        )

    else:

        _print_human(
            bundle,
            max_chars=(
                args.max_chars
            ),
        )


if __name__ == "__main__":
    main()
