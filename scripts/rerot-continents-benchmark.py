#!/usr/bin/env python3
"""End-to-end benchmark and quality verification script for RERoT multi-continent generation.

Prompt: "世界上每个大洲有哪些国家"
Configuration: temperature=0, seed=424242, rerot=true, rerot_trace=true.
Monitors /metrics:
  - rerot_parallel_model_tokens
  - rerot_parallel_seconds
  - computes decode phase parallel_tps = Δrerot_parallel_model_tokens / Δrerot_parallel_seconds
Quality verification:
  1. Parses root-level <ol>
  2. Verifies 8-continent subtask structure (exactly 8 subtasks)
  3. Checks for infinite nesting, repeated headings, and runaway loops
  4. Confirms real country / geographic content for all 8 continents
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


METRICS_KEYS = (
    "rerot_completed_episode_total",
    "rerot_completed_model_tokens",
    "rerot_parallel_model_tokens",
    "rerot_completed_episode_seconds",
    "rerot_parallel_seconds",
    "rerot_public_tokens",
    "rerot_private_tokens",
    "rerot_pending_tokens",
    "rerot_hard_aborts",
    "rerot_final_fences",
    "rerot_people_capacity",
    "rerot_people_resident",
    "rerot_pens_capacity",
    "rerot_pens_allocated",
    "rerot_pens_running",
    "rerot_batch_people",
    "rerot_batch_pens",
    "rerot_frontier_rows",
    "rerot_brain_bytes",
    "rerot_hand_bytes",
)

CONTINENT_MAP: dict[str, dict[str, Any]] = {
    "亚洲": {
        "aliases": ["亚洲", "亚细亚洲", "asia"],
        "keywords": [
            "中国", "日本", "韩国", "朝鲜", "蒙古", "印度", "巴基斯坦", "孟加拉国",
            "孟加拉", "越南", "老挝", "柬埔寨", "泰国", "缅甸", "马来西亚", "新加坡",
            "印度尼西亚", "文莱", "菲律宾", "东帝汶", "尼泊尔", "不丹", "马尔代夫",
            "斯里兰卡", "阿富汗", "伊拉克", "伊朗", "叙利亚", "约旦", "黎巴嫩", "以色列",
            "巴勒斯坦", "沙特阿拉伯", "沙特", "也门", "阿曼", "阿联酋", "卡塔尔",
            "科威特", "巴林", "土耳其", "塞浦路斯", "哈萨克斯坦", "乌兹别克斯坦",
            "土库曼斯坦", "吉尔吉斯斯坦", "塔吉克斯坦", "格鲁吉亚", "阿塞拜疆", "亚美尼亚"
        ],
    },
    "欧洲": {
        "aliases": ["欧洲", "欧罗巴洲", "europe"],
        "keywords": [
            "英国", "法国", "德国", "意大利", "西班牙", "葡萄牙", "荷兰", "比利时",
            "卢森堡", "爱尔兰", "瑞士", "奥地利", "瑞典", "挪威", "芬兰", "丹麦",
            "冰岛", "波兰", "捷克", "斯洛伐克", "匈牙利", "希腊", "俄罗斯", "乌克兰",
            "白俄罗斯", "罗马尼亚", "保加利亚", "塞尔维亚", "克罗地亚", "斯洛文尼亚",
            "波黑", "黑山", "阿尔巴尼亚", "北马其顿", "爱沙尼亚", "拉脱维亚", "立陶宛",
            "摩尔多瓦", "马耳他", "摩纳哥", "安道尔", "圣马力诺", "梵蒂冈", "列支敦士登"
        ],
    },
    "非洲": {
        "aliases": ["非洲", "阿非利加洲", "africa"],
        "keywords": [
            "埃及", "苏丹", "南苏丹", "利比亚", "突尼斯", "阿尔及利亚", "摩洛哥",
            "埃塞俄比亚", "厄立特里亚", "索马里", "吉布提", "肯尼亚", "坦桑尼亚",
            "乌干达", "卢旺达", "布隆迪", "塞舌尔", "尼日利亚", "加纳", "科特迪瓦",
            "塞内加尔", "马里", "尼日尔", "几内亚", "利比里亚", "塞拉利昂", "多哥",
            "贝宁", "冈比亚", "佛得角", "南非", "津巴布韦", "赞比亚", "安哥拉",
            "纳米比亚", "博茨瓦纳", "莫桑比克", "马达加斯加", "毛里求斯", "刚果",
            "喀麦隆", "加蓬", "乍得", "中非"
        ],
    },
    "北美洲": {
        "aliases": ["北美洲", "北美", "north america"],
        "keywords": [
            "美国", "加拿大", "墨西哥", "危地马拉", "伯利兹", "萨尔瓦多", "洪都拉斯",
            "尼加拉瓜", "哥斯达黎加", "巴拿马", "古巴", "牙买加", "海地", "多米尼加",
            "巴哈马", "巴巴多斯", "特立尼达和多巴哥", "格陵兰", "圣卢西亚"
        ],
    },
    "南美洲": {
        "aliases": ["南美洲", "南美", "south america"],
        "keywords": [
            "巴西", "阿根廷", "智利", "哥伦比亚", "秘鲁", "委内瑞拉", "厄瓜多尔",
            "玻利维亚", "巴拉圭", "乌拉圭", "圭亚那", "苏里南", "法属圭亚那"
        ],
    },
    "大洋洲": {
        "aliases": ["大洋洲", "oceania", "australasia"],
        "keywords": [
            "澳大利亚", "新西兰", "巴布亚新几内亚", "斐济", "所罗门群岛", "瓦努阿图",
            "萨摩亚", "汤加", "密克罗尼西亚", "帕劳", "马绍尔群岛", "基里巴斯", "瑙鲁",
            "图瓦卢"
        ],
    },
    "南极洲": {
        "aliases": ["南极洲", "南极", "antarctica"],
        "keywords": [
            "无常住", "无国家", "条约", "科考站", "企鹅", "科学考察", "领土主权",
            "南极条约", "没有国家", "无人定居", "非主权", "科研站", "无独立主权国家",
            "不属于任何国家", "无原住民", "公海", "国际条约", "中山站", "长城站",
            "昆仑站", "泰山站", "秦岭站", "麦克默多"
        ],
    },
    "西兰大陆/第八大洲": {
        "aliases": [
            "西兰大陆", "西兰蒂亚", "zealandia", "第八大洲", "第八洲", "第八大陆",
            "中美洲", "拉丁美洲", "加勒比"
        ],
        "keywords": [
            "新西兰", "新喀里多尼亚", "淹没", "水下", "第七大洲", "第八大洲", "大陆架",
            "危地马拉", "巴拿马", "哥斯达黎加", "古巴", "牙买加", "海地", "大洋洲"
        ],
    },
}


@dataclass
class QualityVerificationResult:
    has_root_ol: bool = False
    root_ol_text: str = ""
    subtask_items: list[str] = field(default_factory=list)
    subtask_count: int = 0
    has_8_subtasks: bool = False

    has_nested_ol: bool = False
    has_repeated_nested_headings: bool = False
    heading_anomalies: list[str] = field(default_factory=list)

    matched_continents: list[str] = field(default_factory=list)
    continent_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    all_8_continents_have_content: bool = False

    content_length: int = 0
    reasoning_length: int = 0
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)


def request_json(
    url: str,
    payload: dict[str, Any],
    api_key: str = "",
    timeout: float = 900.0,
) -> tuple[float, dict[str, Any]]:
    """Send HTTP POST request with JSON payload and return wall time and decoded JSON."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        err_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {err_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Connection failed to {url}: {error.reason}") from error

    elapsed = time.monotonic() - started
    try:
        result = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Failed to decode server JSON response: {error}") from error
    return elapsed, result


def fetch_metrics(base_url: str, api_key: str = "", timeout: float = 30.0) -> dict[str, float]:
    """Scrape Prometheus metrics from /metrics endpoint."""
    url = f"{base_url.rstrip('/')}/metrics"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except Exception as error:
        raise RuntimeError(f"Failed to scrape metrics from {url}: {error}") from error

    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:") or "{" in line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                values[parts[0].removeprefix("llamacpp:")] = float(parts[1])
            except ValueError:
                continue
    return values


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    """Calculate metric delta, defaulting to 0.0 if not found."""
    return after.get(name, 0.0) - before.get(name, 0.0)


def extract_root_ol(text: str) -> tuple[Optional[str], list[str]]:
    """Extract root-level <ol> and its <li> children."""
    match = re.search(r"<ol\b[^>]*>(.*?)</ol>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None, []
    ol_inner = match.group(1)

    items: list[str] = []
    # Primary: match closed <li>...</li>
    closed_items = re.findall(r"<li\b[^>]*>(.*?)</li>", ol_inner, re.DOTALL | re.IGNORECASE)
    if closed_items:
        items = [re.sub(r"<[^>]+>", "", item).strip() for item in closed_items]
    else:
        # Fallback: unclosed <li> splits
        splits = re.split(r"<li\b[^>]*>", ol_inner, flags=re.IGNORECASE)
        for split in splits[1:]:
            cleaned = re.sub(r"<[^>]+>", "", split).strip()
            if cleaned:
                items.append(cleaned)
    return match.group(0), items


def check_heading_anomalies(text: str, root_ol_raw: Optional[str]) -> tuple[bool, list[str]]:
    """Check for nested <ol>, nested headings, consecutive duplicate headings, and runaway loops."""
    anomalies: list[str] = []

    # 1. Nested <ol> inside root <ol>
    if root_ol_raw:
        nested_ols = len(re.findall(r"<ol\b", root_ol_raw, re.IGNORECASE))
        if nested_ols > 1:
            anomalies.append(f"Nested <ol> detected inside root <ol> ({nested_ols} <ol> tags found)")

    # 2. Nested heading tags: e.g. <hN> containing another <hM> before </hN>
    nested_h_pattern = re.compile(
        r"<h([1-6])\b[^>]*>(?:(?!</h\1>).)*?<h[1-6]\b",
        re.DOTALL | re.IGNORECASE,
    )
    if nested_h_pattern.search(text):
        anomalies.append("Syntactically nested heading tags (<hN>...<hM>...) detected")

    # 3. Extract all explicit headings (HTML <hN> and markdown #)
    headings: list[str] = []
    for h_match in re.finditer(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]>", text, re.DOTALL | re.IGNORECASE):
        cleaned = re.sub(r"<[^>]+>", "", h_match.group(1)).strip()
        if cleaned:
            headings.append(cleaned)

    for md_match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        cleaned = md_match.group(2).strip()
        if cleaned:
            headings.append(cleaned)

    # 4. Check consecutive duplicates (stuttering loop)
    for i in range(len(headings) - 1):
        h1 = re.sub(r"\s+", "", headings[i].lower())
        h2 = re.sub(r"\s+", "", headings[i + 1].lower())
        if h1 and h1 == h2:
            anomalies.append(f"Consecutive identical heading repeated: '{headings[i]}'")

    # 5. Check excessive duplicate headings across the entire document
    heading_freq: dict[str, int] = {}
    for h in headings:
        key = re.sub(r"\s+", "", h.lower())
        if key:
            heading_freq[key] = heading_freq.get(key, 0) + 1
    for key, count in heading_freq.items():
        if count > 4:
            anomalies.append(f"Heading repeated excessively ({count} times): key='{key}'")

    # 6. Check excessive heading count explosion (runaway planner/worker heading generation)
    if len(headings) > 40:
        anomalies.append(f"Excessive total heading count: {len(headings)} headings detected")

    # 7. Check for tight repeating phrase loops inside headings (e.g. '亚洲亚洲亚洲')
    for h in headings:
        cleaned = re.sub(r"\s+", "", h)
        if len(cleaned) >= 6:
            for sub_len in (2, 3, 4):
                for start in range(len(cleaned) - sub_len * 3 + 1):
                    unit = cleaned[start:start + sub_len]
                    if cleaned[start:].startswith(unit * 3):
                        anomalies.append(f"Stuttering phrase loop detected in heading '{h}': '{unit}' x 3+")
                        break

    return len(anomalies) > 0, anomalies


def verify_continents_quality(
    content: str,
    reasoning_content: str,
) -> QualityVerificationResult:
    """Verify response quality against all required continents criteria."""
    result = QualityVerificationResult()
    result.content_length = len(content)
    result.reasoning_length = len(reasoning_content)
    full_text = f"{reasoning_content}\n{content}"

    # 1. Parse root-level <ol>
    root_ol, subtasks = extract_root_ol(full_text)
    result.has_root_ol = root_ol is not None
    result.root_ol_text = root_ol or ""
    result.subtask_items = subtasks
    result.subtask_count = len(subtasks)

    if not result.has_root_ol:
        result.failure_reasons.append("Root <ol>...</ol> not found in response text")
    elif result.subtask_count == 0:
        result.failure_reasons.append("Root <ol> was found but contained zero <li> items")

    # 2. Verify 8-continent subtask structure
    result.has_8_subtasks = result.subtask_count == 8
    if result.has_root_ol and result.subtask_count != 8:
        result.failure_reasons.append(
            f"Expected exactly 8 continent subtask items, but got {result.subtask_count}: {subtasks}"
        )

    # 3. Check for infinite nesting and repeated runaway headings
    has_anomalies, anomalies = check_heading_anomalies(full_text, root_ol)
    result.has_repeated_nested_headings = has_anomalies
    result.heading_anomalies = anomalies
    if has_anomalies:
        result.failure_reasons.extend(anomalies)

    # 4. Check real content coverage for continents
    # Match subtask items to continents or check direct continent coverage in document
    matched_continents: set[str] = set()
    continent_details: dict[str, dict[str, Any]] = {}

    for continent_name, conf in CONTINENT_MAP.items():
        aliases = conf["aliases"]
        keywords = conf["keywords"]

        # Check if continent was named in subtasks
        in_subtasks = any(
            any(alias in item.lower() for alias in aliases)
            for item in subtasks
        )

        # Check if continent was mentioned in text
        in_text = any(alias in full_text.lower() for alias in aliases)

        # Count keyword (country/feature) matches in full text
        found_keywords = [kw for kw in keywords if kw in full_text]
        keyword_count = len(found_keywords)

        has_substantive_content = (in_text or in_subtasks) and (keyword_count >= 2 or (continent_name == "南极洲" and keyword_count >= 1))

        if has_substantive_content:
            matched_continents.add(continent_name)

        continent_details[continent_name] = {
            "in_subtasks": in_subtasks,
            "in_text": in_text,
            "keyword_match_count": keyword_count,
            "sample_keywords": found_keywords[:6],
            "has_substantive_content": has_substantive_content,
        }

    result.matched_continents = sorted(matched_continents)
    result.continent_details = continent_details

    # All 8 recognized continent / region partitions must have substantive real content
    result.all_8_continents_have_content = len(matched_continents) >= 8
    if not result.all_8_continents_have_content:
        missing = [name for name, d in continent_details.items() if not d["has_substantive_content"]]
        result.failure_reasons.append(
            f"Missing substantive content for {len(missing)} continent(s): {missing} "
            f"(only {len(matched_continents)}/8 covered: {sorted(matched_continents)})"
        )

    # Overall pass
    result.passed = (
        result.has_root_ol
        and result.has_8_subtasks
        and not result.has_repeated_nested_headings
        and result.all_8_continents_have_content
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continents prompt benchmark with RERoT enabled and verify throughput and content quality."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLAMA_API_KEY", ""),
        help="API key for Authorization: Bearer (default: LLAMA_API_KEY env var)",
    )
    parser.add_argument("--model", default="ornith-1.5", help="Model name (default: ornith-1.5)")
    parser.add_argument(
        "--prompt",
        default="世界上每个大洲有哪些国家",
        help="Benchmark prompt (default: '世界上每个大洲有哪些国家')",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--seed", type=int, default=424242, help="Sampling seed (default: 424242)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate (default: 8192)")
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP request timeout in seconds (default: 900)")
    parser.add_argument(
        "--min-parallel-tps",
        type=float,
        default=0.0,
        help="Minimum parallel TPS required to pass (default: 0.0)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON benchmark report")
    parser.add_argument(
        "--response-file",
        type=Path,
        default=None,
        help="Optional saved response JSON file for offline verification without live HTTP calls",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Reasoning effort is low. Keep thinking brief and focused, then conclude.",
            },
            {
                "role": "user",
                "content": args.prompt,
            }
        ],
        "reasoning_effort": "low",
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "stream": False,
        "rerot": True,
        "rerot_trace": True,
    }

    metrics_before: dict[str, float] = {}
    metrics_after: dict[str, float] = {}
    deltas: dict[str, float] = {}
    wall_seconds: float = 0.0
    response_json_data: dict[str, Any] = {}

    if args.response_file:
        print(f"Loading offline response from: {args.response_file}")
        try:
            content_str = args.response_file.read_text(encoding="utf-8")
            response_json_data = json.loads(content_str)
        except Exception as error:
            print(f"ERROR: Failed to read response file {args.response_file}: {error}", file=sys.stderr)
            return 1
        wall_seconds = 0.0
        # If response file contains pre-captured metrics, extract them
        if "metrics_delta" in response_json_data:
            deltas = response_json_data["metrics_delta"]
        elif "rerot" in response_json_data and "metrics_delta" in response_json_data["rerot"]:
            deltas = response_json_data["rerot"]["metrics_delta"]
    else:
        endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
        print(f"Target endpoint: {endpoint}")
        print(f"Prompt: {args.prompt}")
        print(f"Params: temperature={args.temperature}, seed={args.seed}, rerot=True, rerot_trace=True")

        # 1. Scrape metrics before request
        try:
            metrics_before = fetch_metrics(args.base_url, args.api_key, timeout=args.timeout)
        except Exception as error:
            print(f"ERROR: Failed to fetch initial metrics: {error}", file=sys.stderr)
            return 1

        # 2. Dispatch request
        print("Dispatching chat completion request...")
        try:
            wall_seconds, response_json_data = request_json(
                endpoint, payload, api_key=args.api_key, timeout=args.timeout
            )
        except Exception as error:
            print(f"ERROR: Request failed: {error}", file=sys.stderr)
            return 1

        # 3. Scrape metrics after request
        try:
            metrics_after = fetch_metrics(args.base_url, args.api_key, timeout=args.timeout)
        except Exception as error:
            print(f"ERROR: Failed to fetch post-request metrics: {error}", file=sys.stderr)
            return 1

        deltas = {k: metric_delta(metrics_before, metrics_after, k) for k in METRICS_KEYS}

    # Extract response text
    choices = response_json_data.get("choices", [])
    if not choices:
        print("ERROR: Response choices is empty", file=sys.stderr)
        return 1

    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or ""

    # Timing metrics
    timings = response_json_data.get("timings", {})
    predicted_n = float(timings.get("predicted_n", 0))
    predicted_ms = float(timings.get("predicted_ms", 0))
    serial_tps = (predicted_n / (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0

    # Parallel throughput calculation
    delta_parallel_tokens = deltas.get("rerot_parallel_model_tokens", 0.0)
    delta_parallel_seconds = deltas.get("rerot_parallel_seconds", 0.0)
    parallel_tps = (
        delta_parallel_tokens / delta_parallel_seconds
        if delta_parallel_seconds > 0
        else 0.0
    )

    # Request-wide aggregate throughput calculation
    delta_completed_tokens = deltas.get("rerot_completed_model_tokens", 0.0)
    delta_completed_seconds = deltas.get("rerot_completed_episode_seconds", 0.0)
    aggregate_tps = (
        delta_completed_tokens / delta_completed_seconds
        if delta_completed_seconds > 0
        else 0.0
    )

    # Quality verification
    quality = verify_continents_quality(content, reasoning_content)

    # System invariant checks
    checks: dict[str, bool] = {
        "root_ol_parsed": quality.has_root_ol,
        "eight_continents_subtasks": quality.has_8_subtasks,
        "no_infinite_nested_repeated_headings": not quality.has_repeated_nested_headings,
        "eight_continents_real_content": quality.all_8_continents_have_content,
    }

    if not args.response_file:
        checks["one_completed_episode"] = deltas.get("rerot_completed_episode_total", 0.0) >= 1.0
        checks["no_hard_aborts"] = deltas.get("rerot_hard_aborts", 0.0) == 0.0
        checks["one_final_fence"] = deltas.get("rerot_final_fences", 0.0) >= 1.0
        checks["positive_parallel_tps"] = parallel_tps > 0.0
        if args.min_parallel_tps > 0.0:
            checks["meets_min_parallel_tps"] = parallel_tps >= args.min_parallel_tps

    all_passed = all(checks.values()) and quality.passed

    # Console Output Summary
    print("\n" + "=" * 64)
    print("           RERoT Continents Benchmark Report")
    print("=" * 64)
    print(f"Overall Result:      {'PASS' if all_passed else 'FAIL'}")
    print(f"Prompt:              {args.prompt}")
    print(f"Client Wall Time:    {wall_seconds:.3f} s")
    print("\n--- Decode Throughput Metrics ---")
    print(f"Parallel Model Toks: {delta_parallel_tokens:,.0f}")
    print(f"Parallel Seconds:    {delta_parallel_seconds:.3f} s")
    print(f"Parallel TPS:        {parallel_tps:.3f} tok/s")
    print(f"Aggregate Model Toks:{delta_completed_tokens:,.0f}")
    print(f"Episode Seconds:     {delta_completed_seconds:.3f} s")
    print(f"Aggregate TPS:       {aggregate_tps:.3f} tok/s")
    if serial_tps > 0:
        print(f"Serial Timing TPS:   {serial_tps:.3f} tok/s ({predicted_n:.0f} toks / {predicted_ms:.1f} ms)")

    print("\n--- Content Quality Verification ---")
    print(f"Root <ol> Present:   {'YES' if quality.has_root_ol else 'NO'} ({quality.subtask_count} items)")
    print(f"8 Subtasks Valid:    {'YES' if quality.has_8_subtasks else 'NO'}")
    if quality.subtask_items:
        print(f"Subtask Items:       {quality.subtask_items}")
    print(f"Heading Anomalies:   {len(quality.heading_anomalies)} anomalies")
    for anom in quality.heading_anomalies:
        print(f"  * {anom}")
    print(f"Continents Covered:  {len(quality.matched_continents)}/8 ({quality.matched_continents})")
    print(f"8 Continents Real:   {'YES' if quality.all_8_continents_have_content else 'NO'}")

    print("\n--- Individual Verification Checks ---")
    for check_name, check_ok in checks.items():
        mark = "✓" if check_ok else "✗"
        status = "PASS" if check_ok else "FAIL"
        print(f"  [{mark}] {check_name}: {status}")

    if quality.failure_reasons:
        print("\n--- Failure Reasons ---")
        for reason in quality.failure_reasons:
            print(f"  - {reason}")
    print("=" * 64 + "\n")

    # Construct report object
    report = {
        "schema_version": 1,
        "passed": all_passed,
        "request": payload,
        "wall_seconds": wall_seconds,
        "throughput": {
            "parallel_tokens": delta_parallel_tokens,
            "parallel_seconds": delta_parallel_seconds,
            "parallel_tps": parallel_tps,
            "aggregate_tokens": delta_completed_tokens,
            "aggregate_seconds": delta_completed_seconds,
            "aggregate_tps": aggregate_tps,
            "serial_tps": serial_tps,
        },
        "quality": asdict(quality),
        "checks": checks,
        "metrics_deltas": deltas,
    }

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Full benchmark report written to: {args.output}")
        except Exception as error:
            print(f"WARNING: Failed to write output file: {error}", file=sys.stderr)

    return 0 if all_passed else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
