#!/usr/bin/env python3
"""Check ordinary vs probe->simple on an explicitly supplied, already-running server.

No model is loaded and no service is started or stopped by this script. Run only
against a server whose model, backend and resource limits have been checked,
with support for two completions per request.
"""

import argparse
import json
import urllib.error
import urllib.request


def request(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error


def completion(url, payload, timeout):
    with request(url, payload, timeout) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def compare_simple(url, payload, timeout, label):
    ordinary = completion(url, dict(payload, rerot=False), timeout)
    routed = completion(url, dict(payload, rerot=True), timeout)
    usage = routed["usage"]["rerot"]
    if usage["probe_tokens"] <= 0 or usage["frame_tokens"] != 0:
        raise AssertionError(f"{label}: did not exercise probe->simple")
    if ordinary["choices"] != routed["choices"]:
        raise AssertionError(
            f"{label}: choices differ\nordinary={ordinary['choices']}\nrouted={routed['choices']}"
        )
    sampled = ordinary["usage"]["completion_tokens"]
    if routed["usage"]["completion_tokens"] != sampled or usage["sampled_tokens"] != sampled:
        raise AssertionError(f"{label}: probe polluted ordinary completion accounting")
    if ordinary["choices"][0]["finish_reason"] != "stop":
        raise AssertionError(f"{label}: ordinary baseline did not finish naturally")
    print(f"PASS {label}: sampled={sampled}, probe={usage['probe_tokens']}")
    return ordinary


def compare_stream(url, payload, timeout, ordinary):
    merged = {"content": "", "reasoning_content": ""}
    finishes = []
    done = 0
    with request(url, dict(payload, rerot=True, stream=True), timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done += 1
                continue
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                for key in merged:
                    merged[key] += delta.get(key) or ""
                if delta.get("tool_calls"):
                    raise AssertionError("simple stream leaked an internal tool call")
                if choice.get("finish_reason") is not None:
                    finishes.append(choice["finish_reason"])
    message = ordinary["choices"][0]["message"]
    if any(merged[key] != (message.get(key) or "") for key in merged):
        raise AssertionError(f"stream reasoning/content differs: {merged}")
    if finishes != ["stop"] or done != 1:
        raise AssertionError(f"stream terminal events differ: finishes={finishes}, DONE={done}")
    print("PASS streaming: matching channels, one stop and one DONE")


def compare_multiple(url, payload, timeout, ordinary):
    result = completion(url, dict(payload, rerot=True, n=2), timeout)
    expected = ordinary["choices"][0]["message"]
    if len(result["choices"]) != 2 or {choice["index"] for choice in result["choices"]} != {0, 1}:
        raise AssertionError("two-completion response lost or duplicated an index")
    if any(choice["message"] != expected or choice["finish_reason"] != "stop"
           for choice in result["choices"]):
        raise AssertionError("two-completion response changed ordinary continuation")
    sampled = 2 * ordinary["usage"]["completion_tokens"]
    prompt = ordinary["usage"]["prompt_tokens"]
    usage = result["usage"]
    if (usage["completion_tokens"] != sampled or usage["prompt_tokens"] != prompt
            or usage["total_tokens"] != prompt + sampled
            or usage["rerot"]["sampled_tokens"] != sampled):
        raise AssertionError(f"two-completion aggregate usage is wrong: {usage}")

    messages = {i: {"content": "", "reasoning_content": ""} for i in (0, 1)}
    finishes = []
    usages = []
    done = 0
    with request(url, dict(payload, rerot=True, n=2, stream=True,
                           stream_options={"include_usage": True}), timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done += 1
                continue
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage") is not None:
                usages.append(event["usage"])
            for choice in event.get("choices", []):
                index = choice["index"]
                for key in messages[index]:
                    messages[index][key] += choice.get("delta", {}).get(key) or ""
                if choice.get("finish_reason") is not None:
                    finishes.append((index, choice["finish_reason"]))
    if sorted(finishes) != [(0, "stop"), (1, "stop")] or done != 1 or len(usages) != 1:
        raise AssertionError(f"two-completion stream lifecycle: {finishes}, DONE={done}, usages={len(usages)}")
    if any(message[key] != (expected.get(key) or "")
           for message in messages.values() for key in message):
        raise AssertionError("two-completion stream changed reasoning/content")
    if (usages[0]["completion_tokens"] != sampled
            or usages[0]["prompt_tokens"] != prompt
            or usages[0]["rerot"]["sampled_tokens"] != sampled
            or usages[0]["rerot"]["probe_tokens"] != usage["rerot"]["probe_tokens"]):
        raise AssertionError(f"two-completion stream aggregate usage is wrong: {usages[0]}")
    print(f"PASS two-completion sync/stream: sampled={sampled}, one aggregate usage and DONE")


def compare_input_groups(url, model, timeout):
    base = {"model": model, "temperature": 0, "seed": 424242,
            "max_tokens": 8, "rerot": False, "stream": False}
    prompts = ["1 + 1 =", "2 + 2 ="]
    baselines = [completion(url, dict(base, prompt=prompt), timeout) for prompt in prompts]
    expected_prompt = sum(item["usage"]["prompt_tokens"] for item in baselines)
    expected_sampled = 2 * sum(item["usage"]["completion_tokens"] for item in baselines)
    batch = dict(base, prompt=prompts, n=2)
    result = completion(url, batch, timeout)
    if sorted(choice["index"] for choice in result["choices"]) != [0, 1, 2, 3]:
        raise AssertionError("multi-input response lost completion indices")
    usages = [result["usage"]]
    finishes = []
    done = 0
    with request(url, dict(batch, stream=True, stream_options={"include_usage": True}), timeout) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                done += 1
                continue
            event = json.loads(data)
            if not isinstance(event, dict) or "error" in event:
                raise AssertionError(f"invalid multi-input SSE event: {event}")
            if event.get("usage") is not None:
                usages.append(event["usage"])
            finishes.extend(choice["index"] for choice in event.get("choices", [])
                            if choice.get("finish_reason") is not None)
    if sorted(finishes) != [0, 1, 2, 3] or done != 1 or len(usages) != 2:
        raise AssertionError("multi-input SSE lost finishes or duplicated usage/DONE")
    for usage in usages:
        if (usage["prompt_tokens"] != expected_prompt
                or usage["completion_tokens"] != expected_sampled
                or usage["total_tokens"] != expected_prompt + expected_sampled
                or "rerot" in usage):
            raise AssertionError(f"multi-input usage grouping is wrong: {usage}")
    print(f"PASS multi-input sync/stream: prompt={expected_prompt}, sampled={expected_sampled}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Already-running, resource-checked server URL")
    parser.add_argument("--model", required=True, help="Model alias exposed by that server")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    base = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Which number is larger: 9.11 or 9.9? Answer briefly."}],
        "temperature": 0,
        "seed": 424242,
        "max_tokens": 256,
        "stream": False,
        "reasoning_budget_tokens": 32,
    }
    ordinary = compare_simple(url, base, args.timeout, "greedy-budget")
    compare_simple(url, dict(base, temperature=0.7, seed=9317,
                            reasoning_budget_tokens=16, logprobs=True,
                            top_logprobs=3, post_sampling_probs=True),
                   args.timeout, "seeded-logprobs")
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "comparison", "strict": True,
            "schema": {
                "type": "object",
                "properties": {"larger": {"type": "string", "enum": ["9.11", "9.9"]}},
                "required": ["larger"], "additionalProperties": False,
            },
        },
    }
    json_result = compare_simple(url, dict(base, reasoning_budget_tokens=16, response_format=schema),
                                 args.timeout, "user-json-grammar")
    if json.loads(json_result["choices"][0]["message"]["content"]) != {"larger": "9.9"}:
        raise AssertionError("comparison result is incorrect")
    compare_stream(url, base, args.timeout, ordinary)
    compare_multiple(url, base, args.timeout, ordinary)
    compare_input_groups(args.base_url.rstrip("/") + "/v1/completions", args.model, args.timeout)
    print("All continuation/accounting checks passed; DAG execution was not tested.")


if __name__ == "__main__":
    main()
