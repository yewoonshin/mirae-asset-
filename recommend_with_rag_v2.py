from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


# =========================================================
# 파일 자동 탐색
# =========================================================

def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []

    for p in paths:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)

        if key not in seen:
            seen.add(key)
            out.append(p)

    return out


def _search_roots() -> list[Path]:
    roots = [
        SCRIPT_DIR,
        Path.cwd(),
    ]

    if SCRIPT_DIR.parent != SCRIPT_DIR:
        roots.append(SCRIPT_DIR.parent)

    return _unique_paths(roots)


def find_file(
    filename: str,
    explicit_path: str | None = None,
) -> Path:
    if explicit_path:
        p = Path(explicit_path)

        if not p.exists():
            raise FileNotFoundError(
                f"{filename}을 찾을 수 없습니다.\n"
                f"path={p}"
            )

        return p.resolve()

    direct = SCRIPT_DIR / filename

    if direct.exists():
        return direct.resolve()

    cwd_direct = Path.cwd() / filename

    if cwd_direct.exists():
        return cwd_direct.resolve()

    matches: list[Path] = []

    for root in _search_roots():
        try:
            matches.extend(
                p
                for p in root.rglob(filename)
                if p.is_file()
            )
        except (PermissionError, OSError):
            continue

    matches = _unique_paths(matches)

    if not matches:
        raise FileNotFoundError(
            f"{filename}을 자동으로 찾지 못했습니다.\n"
            f"검색 시작 위치: {[str(x) for x in _search_roots()]}"
        )

    under_script = [
        p
        for p in matches
        if SCRIPT_DIR in p.parents
    ]

    candidates = under_script or matches

    if len(candidates) == 1:
        return candidates[0].resolve()

    processed = [
        p
        for p in candidates
        if "_processed" in p.parts
    ]

    if len(processed) == 1:
        return processed[0].resolve()

    shown = "\n".join(
        f"  - {p}"
        for p in candidates[:20]
    )

    raise RuntimeError(
        f"{filename} 후보가 여러 개라 자동 선택할 수 없습니다.\n"
        f"{shown}\n"
        "해당 파일 경로를 CLI 옵션으로 직접 지정하세요."
    )


# =========================================================
# nl_recommend_v2.py 실행
# =========================================================

def _decode_windows_output(data: bytes) -> str:
    """
    Windows PowerShell/콘솔 환경에서 자식 Python 프로세스 출력이
    UTF-8 또는 CP949 중 어느 쪽으로 나와도 안전하게 복원한다.
    """
    if not data:
        return ""

    # 1) UTF-8 strict
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # 2) 한국어 Windows 기본 코드페이지
    try:
        return data.decode("cp949")
    except UnicodeDecodeError:
        pass

    # 3) 마지막 fallback
    return data.decode("utf-8", errors="replace")


def run_nl_recommend_v2(
    nl_file: Path,
    user_query: str,
) -> str:
    cmd = [
        sys.executable,
        str(nl_file),
        user_query,
    ]

    # text=True + encoding="utf-8"로 고정하면
    # Windows에서 CP949 출력이 들어올 때 한글이 깨질 수 있으므로
    # bytes로 받은 뒤 직접 decode한다.
    result = subprocess.run(
        cmd,
        cwd=str(nl_file.parent),
        capture_output=True,
        text=False,
    )

    stdout = _decode_windows_output(
        result.stdout
    )

    stderr = _decode_windows_output(
        result.stderr
    )

    if result.returncode != 0:
        raise RuntimeError(
            "nl_recommend_v2.py 실행 실패\n\n"
            f"[stdout]\n{stdout}\n\n"
            f"[stderr]\n{stderr}"
        )

    return stdout


# =========================================================
# nl_recommend_v2.py 출력 파싱
# =========================================================

RECOMMEND_HEADER_RE = re.compile(
    r"^\[(\d+)\]\s+(.+?)\s*$"
)


def _parse_bool(text: str) -> bool | None:
    s = text.strip().lower()

    if s in {"true", "1", "yes", "y"}:
        return True

    if s in {"false", "0", "no", "n"}:
        return False

    return None


def _parse_float_prefix(text: str) -> float | None:
    m = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )

    if not m:
        return None

    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_first_json_object(
    text: str,
) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()

    for idx, ch in enumerate(text):
        if ch != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(
                text[idx:]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            if (
                "intent" in obj
                or "account_type" in obj
                or "product_type" in obj
            ):
                return obj

    return None


def parse_recommendations(
    stdout: str,
) -> list[dict[str, Any]]:
    lines = stdout.splitlines()

    starts: list[tuple[int, int, str]] = []

    for i, line in enumerate(lines):
        m = RECOMMEND_HEADER_RE.match(
            line.strip()
        )

        if not m:
            continue

        starts.append(
            (
                i,
                int(m.group(1)),
                m.group(2).strip(),
            )
        )

    recommendations: list[dict[str, Any]] = []

    known_labels = {
        "운용사": "asset_manager",
        "상품유형": "product_type",
        "위험등급": "risk_grade",
        "class": "class_code",
        "채널": "channel",
        "IRP 명시": "irp_explicit",
        "비교비용": "comparison_cost_raw",
        "선정근거": "selection_reason",
        "비용 근거 페이지": "fee_source_page",
        "가입자격 근거 페이지": "eligibility_source_page",
        "source_folder_id": "source_folder_id",
        "fund_code": "fund_code",
    }

    for pos, (
        start_idx,
        rank,
        fund_name,
    ) in enumerate(starts):

        end_idx = (
            starts[pos + 1][0]
            if pos + 1 < len(starts)
            else len(lines)
        )

        item: dict[str, Any] = {
            "rank": rank,
            "fund_name": fund_name,
        }

        for raw_line in lines[
            start_idx + 1 : end_idx
        ]:
            stripped = raw_line.strip()

            if not stripped:
                continue

            for label, key in known_labels.items():
                prefix = f"{label}:"

                if stripped.startswith(prefix):
                    value = stripped[
                        len(prefix):
                    ].strip()

                    item[key] = value
                    break

        if "risk_grade" in item:
            val = _parse_float_prefix(
                str(item["risk_grade"])
            )

            if val is not None:
                item["risk_grade"] = int(val)

        if "irp_explicit" in item:
            parsed = _parse_bool(
                str(item["irp_explicit"])
            )

            if parsed is not None:
                item["irp_explicit"] = parsed

        if "comparison_cost_raw" in item:
            raw = str(
                item["comparison_cost_raw"]
            )

            item[
                "comparison_cost_pct"
            ] = _parse_float_prefix(raw)

            basis_match = re.search(
                r"\(([^()]*)\)\s*$",
                raw,
            )

            if basis_match:
                item[
                    "comparison_cost_basis"
                ] = basis_match.group(
                    1
                ).strip()

        recommendations.append(item)

    return recommendations


# =========================================================
# rag_chunks.jsonl 상품 index
# =========================================================

def normalize_fund_name(
    name: str,
) -> str:
    text = unicodedata.normalize(
        "NFKC",
        name or "",
    ).lower()

    return "".join(
        ch
        for ch in text
        if ch.isalnum()
    )


def build_fund_indexes(
    rag_file: Path,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    exact_name: dict[str, set[str]] = {}
    normalized_name: dict[str, set[str]] = {}
    fund_code_index: dict[str, set[str]] = {}

    with rag_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            fund_name = str(
                row.get("fund_name", "")
            ).strip()

            folder_id = str(
                row.get("source_folder_id", "")
            ).strip()

            fund_code = str(
                row.get("fund_code", "")
            ).strip()

            if not folder_id:
                continue

            if fund_name:
                exact_name.setdefault(
                    fund_name,
                    set(),
                ).add(folder_id)

                normalized_name.setdefault(
                    normalize_fund_name(
                        fund_name
                    ),
                    set(),
                ).add(folder_id)

            if fund_code:
                fund_code_index.setdefault(
                    fund_code,
                    set(),
                ).add(folder_id)

    return (
        exact_name,
        normalized_name,
        fund_code_index,
    )


def resolve_source_folder_id(
    recommendation: dict[str, Any],
    exact_name_index: dict[str, set[str]],
    normalized_name_index: dict[str, set[str]],
    fund_code_index: dict[str, set[str]],
) -> str:
    # 1순위: v2 출력에 source_folder_id가 이미 있다면 그대로 사용
    explicit_folder = str(
        recommendation.get(
            "source_folder_id",
            "",
        )
    ).strip()

    if explicit_folder:
        return explicit_folder

    # 2순위: fund_code가 있고 유일하면 사용
    fund_code = str(
        recommendation.get(
            "fund_code",
            "",
        )
    ).strip()

    if fund_code:
        code_matches = fund_code_index.get(
            fund_code,
            set(),
        )

        if len(code_matches) == 1:
            return next(iter(code_matches))

    # 3순위: 상품명 exact
    fund_name = str(
        recommendation["fund_name"]
    ).strip()

    exact_matches = exact_name_index.get(
        fund_name,
        set(),
    )

    if len(exact_matches) == 1:
        return next(iter(exact_matches))

    # 4순위: Unicode/공백/기호 정규화 상품명
    norm = normalize_fund_name(
        fund_name
    )

    norm_matches = normalized_name_index.get(
        norm,
        set(),
    )

    if len(norm_matches) == 1:
        return next(iter(norm_matches))

    if not norm_matches:
        raise KeyError(
            "추천 상품명을 rag_chunks.jsonl에서 "
            "찾지 못했습니다.\n"
            f"fund_name={fund_name}"
        )

    raise ValueError(
        "추천 상품명이 여러 source_folder_id에 "
        "매칭됩니다.\n"
        f"fund_name={fund_name}\n"
        f"matches={sorted(norm_matches)}"
    )


# =========================================================
# product_rag.py import
# =========================================================

def load_product_rag_module(
    product_rag_file: Path,
):
    spec = importlib.util.spec_from_file_location(
        "product_rag_runtime",
        str(product_rag_file),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "product_rag.py를 import할 수 없습니다: "
            f"{product_rag_file}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    if not hasattr(
        module,
        "ProductRAG",
    ):
        raise AttributeError(
            "product_rag.py에 ProductRAG 클래스가 없습니다."
        )

    return module



# =========================================================
# 명시 키워드 deterministic 적합성 검증
# =========================================================

EXPLICIT_KEYWORD_FALLBACKS = {
    "단기채": "단기",
    "단기 채권": "단기",
    "국공채": "국공채",
    "회사채": "회사채",
    "크레딧": "크레딧",
    "고배당": "고배당",
    "미국": "미국",
}


def extract_explicit_keywords(
    user_query: str,
    parsed_conditions: dict[str, Any] | None,
) -> list[str]:
    """
    LLM이 반환한 keywords + 사용자가 실제 입력한 명시 표현만 사용한다.
    추론으로 새로운 선호를 만들지 않는다.
    """
    out: list[str] = []
    seen: set[str] = set()

    if parsed_conditions:
        raw = parsed_conditions.get("keywords", [])

        if isinstance(raw, list):
            for item in raw:
                kw = str(item).strip()

                if kw and kw not in seen:
                    seen.add(kw)
                    out.append(kw)

    q = unicodedata.normalize(
        "NFKC",
        user_query or "",
    ).lower()

    for phrase, kw in EXPLICIT_KEYWORD_FALLBACKS.items():
        if phrase.lower() in q and kw not in seen:
            seen.add(kw)
            out.append(kw)

    return out


def _evidence_text_by_type(
    evidence: list[dict[str, Any]],
    evidence_types: set[str],
) -> str:
    parts: list[str] = []

    for ev in evidence:
        ev_type = (
            ev.get("evidence_type")
            or ev.get("section")
            or ""
        )

        if ev_type in evidence_types:
            parts.append(
                str(ev.get("text", ""))
            )

    return " ".join(parts)


def _norm_for_keyword(text: str) -> str:
    return unicodedata.normalize(
        "NFKC",
        text or "",
    ).lower()


def _extract_pct_rules(
    text: str,
    term: str,
) -> tuple[list[float], list[float]]:
    """
    term과 '같은 짧은 구문' 안에 있는 N% 이상/이하만 연결한다.

    예:
      "장기증권 ... 60%이상, ... 단기증권 ... 40%이하"
    에서 60%를 '단기'와 잘못 연결하지 않도록
    쉼표/문장부호를 넘어가며 역방향 매칭하지 않는다.
    """
    s = _norm_for_keyword(text)

    above: list[float] = []
    below: list[float] = []

    # term 뒤 50자 이내에서만 비율을 찾고,
    # 쉼표/세미콜론/줄바꿈을 넘어가지 않는다.
    forward = re.compile(
        rf"{re.escape(term)}"
        rf"[^,;\n。.!?]{{0,50}}?"
        rf"(\d+(?:\.\d+)?)\s*%\s*(이상|이하)",
        flags=re.IGNORECASE,
    )

    for m in forward.finditer(s):
        value = float(m.group(1))
        direction = m.group(2)

        if direction == "이상":
            above.append(value)
        else:
            below.append(value)

    return above, below


def evaluate_keyword_match(
    *,
    keyword: str,
    recommendation: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    True  = 명시 키워드와 일치하는 근거가 충분함
    False = 반대 근거가 있거나 근거가 부족함

    금융 추천에서 '근거 없음'을 억지로 적합으로 해석하지 않는다.
    """
    kw = _norm_for_keyword(keyword).strip()

    fund_name = _norm_for_keyword(
        str(recommendation.get("fund_name", ""))
    )

    strategy = _norm_for_keyword(
        _evidence_text_by_type(
            evidence,
            {"strategy"},
        )
    )

    assets = _norm_for_keyword(
        _evidence_text_by_type(
            evidence,
            {"assets"},
        )
    )

    objective = _norm_for_keyword(
        _evidence_text_by_type(
            evidence,
            {"objective"},
        )
    )

    combined_core = " ".join(
        [fund_name, strategy, assets, objective]
    )

    # -----------------------------------------------------
    # 단기
    # -----------------------------------------------------
    if kw == "단기":
        short_above, short_below = _extract_pct_rules(
            strategy,
            "단기",
        )

        long_above, long_below = _extract_pct_rules(
            strategy,
            "장기",
        )

        # 가장 중요한 반대 근거를 먼저 본다.
        # 예: 장기 60% 이상 + 단기 40% 이하
        if (
            any(x >= 50 for x in long_above)
            and any(x <= 50 for x in short_below)
        ):
            return {
                "keyword": keyword,
                "matched": False,
                "reason": (
                    "투자전략상 장기 관련 비중이 "
                    f"{max(long_above):g}% 이상이고 "
                    "단기 관련 비중은 "
                    f"{min(short_below):g}% 이하"
                ),
            }

        # 상품명에 단기가 명시되어 있으면 기본적으로 강한 근거.
        # 단, 위의 명시적 반대 배분이 있으면 이미 탈락했다.
        if "단기" in fund_name:
            return {
                "keyword": keyword,
                "matched": True,
                "reason": "상품명에 '단기'가 명시됨",
            }

        # 전략상 단기자산/단기 모펀드가 과반 이상
        if any(x >= 50 for x in short_above):
            return {
                "keyword": keyword,
                "matched": True,
                "reason": (
                    "투자전략에서 단기 관련 자산/모펀드 "
                    f"{max(short_above):g}% 이상 비중 확인"
                ),
            }

        return {
            "keyword": keyword,
            "matched": False,
            "reason": (
                "'단기채 위주'라고 판단할 충분한 "
                "상품명/투자전략 근거가 없음"
            ),
        }

    # -----------------------------------------------------
    # 국공채
    # -----------------------------------------------------
    if kw == "국공채":
        matched = "국공채" in combined_core

        return {
            "keyword": keyword,
            "matched": matched,
            "reason": (
                "상품명/투자전략/투자대상에서 국공채 근거 확인"
                if matched
                else "국공채 투자 근거가 확인되지 않음"
            ),
        }

    # -----------------------------------------------------
    # 회사채
    # -----------------------------------------------------
    if kw == "회사채":
        matched = "회사채" in combined_core

        return {
            "keyword": keyword,
            "matched": matched,
            "reason": (
                "상품명/투자전략/투자대상에서 회사채 근거 확인"
                if matched
                else "회사채 투자 근거가 확인되지 않음"
            ),
        }

    # -----------------------------------------------------
    # 크레딧
    # -----------------------------------------------------
    if kw == "크레딧":
        matched = (
            "크레딧" in combined_core
            or "신용" in strategy
            or "회사채" in strategy
        )

        return {
            "keyword": keyword,
            "matched": matched,
            "reason": (
                "크레딧/신용/회사채 전략 근거 확인"
                if matched
                else "크레딧 관련 전략 근거가 확인되지 않음"
            ),
        }

    # -----------------------------------------------------
    # 고배당
    # -----------------------------------------------------
    if kw == "고배당":
        matched = (
            "고배당" in combined_core
            or "배당" in strategy
            or "배당" in assets
        )

        return {
            "keyword": keyword,
            "matched": matched,
            "reason": (
                "배당 관련 투자전략/대상 근거 확인"
                if matched
                else "고배당/배당 관련 근거가 확인되지 않음"
            ),
        }

    # -----------------------------------------------------
    # 미국
    # -----------------------------------------------------
    if kw == "미국":
        matched = "미국" in combined_core

        return {
            "keyword": keyword,
            "matched": matched,
            "reason": (
                "미국 투자 관련 근거 확인"
                if matched
                else "미국 투자 관련 근거가 확인되지 않음"
            ),
        }

    # 모르는 키워드는 억지로 필터하지 않는다.
    return {
        "keyword": keyword,
        "matched": True,
        "reason": "deterministic 필터 규칙 미정의: 기존 추천 순서 유지",
        "rule_defined": False,
    }


def apply_keyword_filter(
    *,
    products: list[dict[str, Any]],
    keywords: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    모든 정의된 명시 키워드를 만족해야 최종 통과(AND).
    규칙 미정의 키워드는 강제 탈락시키지 않는다.
    """
    if not keywords:
        return products, []

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for product in products:
        checks = [
            evaluate_keyword_match(
                keyword=kw,
                recommendation=product,
                evidence=product.get("evidence", []),
            )
            for kw in keywords
        ]

        product["keyword_checks"] = checks

        defined_checks = [
            c
            for c in checks
            if c.get("rule_defined", True)
        ]

        matched = all(
            bool(c.get("matched"))
            for c in defined_checks
        )

        product["keyword_matched"] = matched

        if matched:
            kept.append(product)
        else:
            rejected.append(
                {
                    "original_rank": product.get("rank"),
                    "fund_name": product.get("fund_name"),
                    "keyword_checks": checks,
                }
            )

    # 기존 비용순 상대 순서는 유지하되,
    # 탈락 후 사용자에게 보이는 순위만 연속 번호로 다시 매긴다.
    for new_rank, product in enumerate(
        kept,
        start=1,
    ):
        product["original_rank"] = product.get("rank")
        product["rank"] = new_rank

    return kept, rejected



# =========================================================
# 통합 pipeline
# =========================================================

def integrate(
    *,
    user_query: str,
    nl_file: Path,
    product_rag_file: Path,
    rag_file: Path,
    include_optional: list[str] | None = None,
) -> dict[str, Any]:
    stdout = run_nl_recommend_v2(
        nl_file=nl_file,
        user_query=user_query,
    )

    parsed_conditions = (
        extract_first_json_object(
            stdout
        )
    )

    recommendations = (
        parse_recommendations(
            stdout
        )
    )

    if not recommendations:
        raise RuntimeError(
            "nl_recommend_v2.py 실행은 성공했지만 "
            "추천 상품 블록([1], [2], ...)을 "
            "파싱하지 못했습니다.\n\n"
            f"{stdout}"
        )

    (
        exact_name_index,
        normalized_name_index,
        fund_code_index,
    ) = build_fund_indexes(
        rag_file
    )

    product_rag_module = (
        load_product_rag_module(
            product_rag_file
        )
    )

    rag = product_rag_module.ProductRAG(
        rag_file
    )

    integrated_products: list[
        dict[str, Any]
    ] = []

    for recommendation in recommendations:
        source_folder_id = (
            resolve_source_folder_id(
                recommendation,
                exact_name_index,
                normalized_name_index,
                fund_code_index,
            )
        )

        evidence_bundle = (
            rag.retrieve_product_evidence(
                user_query=user_query,
                source_folder_id=(
                    source_folder_id
                ),
                include_optional=(
                    include_optional or []
                ),
            )
        )

        combined = dict(
            recommendation
        )

        combined[
            "source_folder_id"
        ] = source_folder_id

        combined[
            "evidence"
        ] = evidence_bundle[
            "evidence"
        ]

        integrated_products.append(
            combined
        )

    explicit_keywords = extract_explicit_keywords(
        user_query,
        parsed_conditions,
    )

    filtered_products, rejected_products = (
        apply_keyword_filter(
            products=integrated_products,
            keywords=explicit_keywords,
        )
    )

    requested_top_k = None

    if parsed_conditions:
        requested_top_k = parsed_conditions.get(
            "top_k"
        )

    return {
        "user_request": user_query,
        "parsed_conditions":
            parsed_conditions,
        "explicit_keywords":
            explicit_keywords,
        "requested_top_k":
            requested_top_k,
        "recommended_products":
            filtered_products,
        "rejected_by_keyword":
            rejected_products,
        "raw_nl_output":
            stdout,
    }


# =========================================================
# 출력
# =========================================================

def compact_text(
    text: str,
    max_chars: int,
) -> str:
    s = re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()

    if (
        max_chars > 0
        and len(s) > max_chars
    ):
        return (
            s[:max_chars].rstrip()
            + " ..."
        )

    return s


def print_human(
    result: dict[str, Any],
    *,
    max_chars: int,
) -> None:
    print("\n" + "=" * 88)
    print("추천 + 투자설명서 RAG 통합 결과")
    print("=" * 88)

    conditions = result.get(
        "parsed_conditions"
    )

    if conditions:
        print("\n[구조화 조건]")
        print(
            json.dumps(
                conditions,
                ensure_ascii=False,
                indent=2,
            )
        )

    keywords = result.get(
        "explicit_keywords",
        [],
    )

    if keywords:
        print(
            "\n[명시 키워드 검증] "
            + ", ".join(keywords)
        )

        rejected = result.get(
            "rejected_by_keyword",
            [],
        )

        if rejected:
            print(
                "  키워드 불일치로 제외: "
                f"{len(rejected)}개"
            )

            for item in rejected:
                reasons = "; ".join(
                    str(check.get("reason", ""))
                    for check in item.get(
                        "keyword_checks",
                        []
                    )
                    if not check.get(
                        "matched",
                        True,
                    )
                )

                print(
                    "   - "
                    f"기존 {item.get('original_rank')}위 "
                    f"{item.get('fund_name')} "
                    f"→ 제외 ({reasons})"
                )

        requested_top_k = result.get(
            "requested_top_k"
        )

        actual = len(
            result.get(
                "recommended_products",
                [],
            )
        )

        if (
            isinstance(requested_top_k, int)
            and actual < requested_top_k
        ):
            print(
                "  요청 수량보다 적게 반환: "
                f"{actual}/{requested_top_k}개 "
                "(조건에 맞지 않는 상품을 억지로 채우지 않음)"
            )

    for product in result[
        "recommended_products"
    ]:
        print("\n" + "-" * 88)

        print(
            f"[{product['rank']}] "
            f"{product['fund_name']}"
        )

        print(
            "  source_folder_id: "
            f"{product['source_folder_id']}"
        )

        if product.get("asset_manager"):
            print(
                "  운용사: "
                f"{product['asset_manager']}"
            )

        if product.get("product_type"):
            print(
                "  상품유형: "
                f"{product['product_type']}"
            )

        if product.get("risk_grade") is not None:
            print(
                "  위험등급: "
                f"{product['risk_grade']}"
            )

        if product.get("class_code"):
            print(
                "  class: "
                f"{product['class_code']}"
            )

        if product.get("channel"):
            print(
                "  채널: "
                f"{product['channel']}"
            )

        if product.get("irp_explicit") is not None:
            print(
                "  IRP 명시: "
                f"{product['irp_explicit']}"
            )

        if product.get(
            "comparison_cost_pct"
        ) is not None:
            basis = product.get(
                "comparison_cost_basis",
                "",
            )

            basis_text = (
                f" ({basis})"
                if basis
                else ""
            )

            print(
                "  비교비용: "
                f"{product['comparison_cost_pct']}%"
                f"{basis_text}"
            )

        if product.get(
            "selection_reason"
        ):
            print(
                "  선정근거: "
                f"{product['selection_reason']}"
            )

        checks = product.get(
            "keyword_checks",
            [],
        )

        if checks:
            print("\n  [키워드 적합성]")

            for check in checks:
                status = (
                    "PASS"
                    if check.get("matched")
                    else "FAIL"
                )

                print(
                    "    - "
                    f"{check.get('keyword')}: "
                    f"{status} / "
                    f"{check.get('reason')}"
                )

        print("\n  [RAG evidence]")

        for ev in product.get(
            "evidence",
            []
        ):
            label = (
                ev.get("evidence_type")
                or ev.get("section")
                or "evidence"
            )

            page = (
                ev.get("page_label")
                or ""
            )

            score = ev.get("score")

            print(
                f"    - {label}"
                f" / {page}"
                f" / score={score}"
            )

            print(
                "      "
                + compact_text(
                    ev.get("text", ""),
                    max_chars,
                )
            )


# =========================================================
# main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "nl_recommend_v2.py 추천 결과에 "
            "product_rag.py 투자설명서 근거를 붙이고 "
            "명시 키워드 적합성을 deterministic하게 검증합니다."
        )
    )

    parser.add_argument(
        "query",
        help="사용자 자연어 요청",
    )

    parser.add_argument(
        "--nl-file",
        default=None,
        help=(
            "nl_recommend_v2.py 경로. "
            "생략 시 자동 탐색"
        ),
    )

    parser.add_argument(
        "--product-rag-file",
        default=None,
        help=(
            "product_rag.py 경로. "
            "생략 시 자동 탐색"
        ),
    )

    parser.add_argument(
        "--rag-file",
        default=None,
        help=(
            "rag_chunks.jsonl 경로. "
            "생략 시 하위 폴더까지 자동 탐색"
        ),
    )

    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        choices=[
            "objective",
            "assets",
            "strategy",
            "risk",
            "purchase_redemption",
            "valuation",
            "fees",
            "tax",
        ],
        help=(
            "기본 evidence 외 추가 category. "
            "예: --include fees tax"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="최종 결과를 JSON으로 출력",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=450,
        help=(
            "사람용 출력의 evidence당 "
            "최대 글자 수. 0이면 전체"
        ),
    )

    args = parser.parse_args()

    nl_file = find_file(
        "nl_recommend_v2.py",
        args.nl_file,
    )

    product_rag_file = find_file(
        "product_rag.py",
        args.product_rag_file,
    )

    rag_file = find_file(
        "rag_chunks.jsonl",
        args.rag_file,
    )

    print(
        "[자동 탐색]\n"
        f"  nl_recommend_v2.py = {nl_file}\n"
        f"  product_rag.py      = {product_rag_file}\n"
        f"  rag_chunks.jsonl    = {rag_file}"
    )

    result = integrate(
        user_query=args.query,
        nl_file=nl_file,
        product_rag_file=product_rag_file,
        rag_file=rag_file,
        include_optional=args.include,
    )

    if args.json:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_human(
            result,
            max_chars=args.max_chars,
        )


if __name__ == "__main__":
    main()
