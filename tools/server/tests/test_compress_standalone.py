#!/usr/bin/env python3
"""Exercise /v1/compress with a local model and only Python's standard library.

Example:
    python3 tools/server/tests/test_compress_standalone.py \
        --binary build/bin/llama-server --model models/stories260K.gguf

The model checks request handling and inference plumbing, not compression quality.
The test template includes tool arguments; the built-in chatml template omits them.
The runner starts one server, uses a free local port, and stops it on exit.
"""

import argparse
import copy
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def is_subsequence(shorter, original):
    """Check bytes or characters without assumptions about the selector."""
    next_index = 0
    for item in original:
        if next_index < len(shorter) and shorter[next_index] == item:
            next_index += 1
    return next_index == len(shorter)


def decode_json(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "response contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise AssertionError("response contains a non-JSON number: " + value)

    result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                        parse_constant=reject_constant)

    def check_unicode(value):
        if isinstance(value, str):
            value.encode("utf-8", errors="strict")
        elif isinstance(value, list):
            for item in value:
                check_unicode(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                check_unicode(key)
                check_unicode(item)

    check_unicode(result)
    return result


def encode_json(value, ascii_only=False):
    # Whitespace and escapes are deliberate: a parse-and-dump response must fail
    # the identity and byte-subsequence checks.
    return (" \n" + json.dumps(value, ensure_ascii=ascii_only, indent=2) + "\t\n").encode("utf-8")


@dataclass
class Response:
    status: int
    headers: dict
    raw: bytes

    def json(self):
        return decode_json(self.raw)


class Client:
    def __init__(self, port, timeout):
        self.base_url = "http://127.0.0.1:" + str(port)
        self.timeout = timeout
        # A local acceptance test must not send requests through an HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, raw=None):
        request = urllib.request.Request(
            self.base_url + path,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="GET" if raw is None else "POST",
        )
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return Response(response.status,
                            {key.lower(): value for key, value in response.headers.items()},
                            response.read())

    def json_request(self, path, body):
        response = self.request(path, encode_json(body))
        require(response.status == 200,
                f"{path} returned {response.status}: {response.raw[:800]!r}")
        return response.json()

    def prompt_tokens(self, request):
        prompt = self.json_request("/apply-template", request)["prompt"]
        tokens = self.json_request("/tokenize", {
            "content": prompt, "add_special": True, "parse_special": True,
        })["tokens"]
        return len(tokens)


def request_body(messages, **extra):
    result = {
        "model": "compression-test",
        "messages": messages,
        "max_tokens": 2,
        "temperature": 0,
        "seed": 7,
        "stream": False,
        "compression_test_marker": {"value": "keep 中文", "items": [3, True, None]},
    }
    result.update(extra)
    return result


SYSTEM = {"role": "system", "content": "You are a coding assistant. Preserve the requested behavior."}

# Keep the fixture small enough for the tiny model's context. It must render
# every field that the tool tests intend to compress and count.
CHAT_TEMPLATE = r"""{%- if strict_alternation is defined and strict_alternation %}
{%- set ns = namespace(index=0) %}
{%- for message in messages %}
{%- if message.role == 'user' or (message.role == 'assistant' and not message.tool_calls and not message.function_call) %}
{%- if (message.role == 'user') != (ns.index % 2 == 0) %}
{{- raise_exception('Conversation roles must alternate user/assistant.') }}
{%- endif %}
{%- set ns.index = ns.index + 1 %}
{%- endif %}
{%- endfor %}
{%- endif %}
{%- if tools %}
{{- '<|im_start|>system\n' }}{{- tools | tojson }}{{- '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}{{- message.content }}
{%- elif message.content %}{{- message.content | tojson }}{%- endif %}
{%- if message.tool_calls %}{{- '\n' }}{{- message.tool_calls | tojson }}{%- endif %}
{%- if message.function_call %}{{- '\n' }}{{- message.function_call | tojson }}{%- endif %}
{%- if message.tool_call_id %}{{- '\n' + message.tool_call_id }}{%- endif %}
{%- if message.reasoning_content %}{{- '\n' + message.reasoning_content }}{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}
"""


def tool_definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read or write a source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assert_tool_closure(messages):
    calls = set()
    results = set()
    for message in messages:
        for call in message.get("tool_calls", []):
            call_id = call["id"]
            require(call_id and call_id not in calls, "tool call IDs must stay unique")
            calls.add(call_id)
        if message["role"] == "tool":
            call_id = message["tool_call_id"]
            require(call_id in calls, "tool result lost its preceding call")
            require(call_id not in results, "duplicate tool result")
            results.add(call_id)
    require(calls == results, "a retained tool call lost its result")


def assert_json_string_deletions(before, after):
    """Inner arguments keep their structure; only string values may shrink."""
    require(type(before) is type(after), "argument value type changed")
    if isinstance(before, str):
        require(is_subsequence(after, before), "argument string was rewritten")
    elif isinstance(before, dict):
        require(list(before) == list(after), "argument keys or order changed")
        for key in before:
            assert_json_string_deletions(before[key], after[key])
    elif isinstance(before, list):
        require(len(before) == len(after), "argument array structure changed")
        for original, result in zip(before, after):
            assert_json_string_deletions(original, result)
    else:
        require(before == after, "non-string argument changed")


class Suite:
    def __init__(self, client, context_size):
        self.client = client
        self.context_size = context_size
        self.target = min(1024, context_size // 2)
        self.passed = 0

    def passed_case(self, name):
        self.passed += 1
        print("PASS " + name, flush=True)

    def compress(self, raw, target, original_tokens=None):
        before = decode_json(raw)
        if original_tokens is None:
            original_tokens = self.client.prompt_tokens(before)
        response = self.client.request("/v1/compress?target_tokens=" + str(target), raw)
        require(response.status == 200,
                f"compress returned {response.status}: {response.raw[:1000]!r}")
        require("application/json" in response.headers.get("content-type", ""),
                "compression must return one JSON response, including for stream:true")
        require(is_subsequence(response.raw, raw), "response is not an original-body byte subsequence")
        after = response.json()
        require(isinstance(after, dict) and "messages" in after, "response is not a chat request")
        require({key: value for key, value in after.items() if key != "messages"} ==
                {key: value for key, value in before.items() if key != "messages"},
                "a top-level request field changed")
        require(0 < len(after["messages"]) <= len(before["messages"]), "invalid message count")
        actual_tokens = self.client.prompt_tokens(after)
        require(actual_tokens <= target, f"actual prompt has {actual_tokens} tokens, budget is {target}")
        expected_headers = {
            "x-compression-original-tokens": original_tokens,
            "x-compression-tokens": actual_tokens,
            "x-compression-target-tokens": target,
        }
        for name, expected in expected_headers.items():
            require(name in response.headers, "missing " + name)
            require(int(response.headers[name]) == expected,
                    f"{name}: expected {expected}, got {response.headers[name]}")
        original_system = [item for item in before["messages"] if item["role"] in ("system", "developer")]
        final_system = [item for item in after["messages"] if item["role"] in ("system", "developer")]
        require(original_system == final_system, "system or developer content changed")
        return response, after

    def fit_and_stream(self):
        for stream in (False, True):
            body = request_body([
                SYSTEM,
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "OK.", "tool_calls": None, "function_call": None},
                {"role": "user", "content": "Reply OK. 中文😀"},
            ], stream=stream)
            raw = encode_json(body, ascii_only=stream)
            count = self.client.prompt_tokens(body)
            response, after = self.compress(raw, count + 16, count)
            require(response.raw == raw, "a request that fits must keep its exact original bytes")
            require(after["stream"] is stream, "stream setting changed")
        self.passed_case("fit preserves bytes and nullable fields; stream:true still returns one JSON request")

    def bad_inputs(self):
        raw = encode_json(request_body([{"role": "user", "content": "Hi"}]))
        for target in ("0", "-1", "banana", "1.5", "999999999999999999999999999999999"):
            response = self.client.request("/v1/compress?target_tokens=" + target, raw)
            require(response.status == 400, f"invalid target {target!r} must return 400")
            require("error" in response.json(), "invalid target must return a JSON error")
        invalid_bodies = [
            b'{"messages":[],"messages":[{"role":"user","content":"Hi"}]}',
            b'{"messages":[{"role":"user","content":"one","content":"two"}]}',
            b'{"messages":[{"role":"user","content":"\\ud800"}]}',
            b'{"messages":[{"role":"user","content":"\xc0\x80"}]}',
        ]
        for invalid in invalid_bodies:
            response = self.client.request("/v1/compress?target_tokens=512", invalid)
            require(response.status == 400, "duplicate keys or invalid Unicode must return 400")
            require("error" in response.json(), "invalid JSON must return a JSON error")
        self.passed_case("invalid budgets, duplicate keys, and invalid Unicode return 400")

    def impossible_budget(self):
        body = request_body([SYSTEM, {"role": "user", "content": "Keep this short instruction intact."}])
        response = self.client.request("/v1/compress?target_tokens=1", encode_json(body))
        require(response.status == 422, "a budget below protected content must return 422")
        require("error" in response.json(), "an impossible budget must return a JSON error")
        self.passed_case("unreachable protected-content budget returns 422")

    def long_user_and_reuse(self):
        lines = 256
        while True:
            content = "Inspect the following trace.\n" + "".join(
                f"entry {index % 11}: cache value unchanged; keep reading.\n" for index in range(lines)
            ) + "\nWhich check should run next?"
            body = request_body([SYSTEM, {"role": "user", "content": content}])
            count = self.client.prompt_tokens(body)
            if count > 2 * self.context_size + 128:
                break
            lines *= 2
        raw = encode_json(body)
        response, after = self.compress(raw, self.target, count)
        require(len(after["messages"]) == 2, "the latest user turn must remain")
        require(0 < len(after["messages"][1]["content"]) < len(content), "large user content did not shrink")
        require(is_subsequence(after["messages"][1]["content"], content), "user text was rewritten")
        second, _ = self.compress(raw, self.target, count)
        require(second.raw == response.raw, "repeated identical requests must select the same bytes")
        reused = self.client.request("/v1/chat/completions", response.raw)
        require(reused.status == 200, f"compressed request cannot run: {reused.raw[:1000]!r}")
        result = reused.json()
        require(result.get("choices") and "message" in result["choices"][0], "completion response is missing")
        require(result.get("usage", {}).get("completion_tokens", 0) > 0, "completion produced no tokens")
        self.passed_case(f"single user over 2×context ({count} tokens), deterministic deletion, and completion reuse")

    def unicode_and_escapes(self):
        content = "Inspect these bytes.\n" + (
            '中文 café e\u0301 😀 \\ path "quoted" / row\tvalue\n' * 192
        ) + "\nReport the result."
        body = request_body([SYSTEM, {"role": "user", "content": content}])
        raw = encode_json(body).replace("😀".encode("utf-8"), b"\\ud83d\\ude00")
        response, after = self.compress(raw, self.target)
        require(len(response.raw) < len(raw), "Unicode case did not compress")
        result = after["messages"][1]["content"]
        require(is_subsequence(result, content), "Unicode text was rewritten")
        result.encode("utf-8", errors="strict")
        self.passed_case("literal UTF-8, surrogate escapes, quotes, and backslashes stay valid")

    def long_read_result(self):
        content = "src/cache.cpp\n" + "".join(
            f"line {index}: if (entry.ready) return cached_value;\n" for index in range(256)
        ) + "end of source\n"
        call = tool_call("read_latest", "read", '{ "path" : "src/cache.cpp" }')
        messages = [SYSTEM, {"role": "user", "content": "Read the file and inspect its cache behavior."},
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "read_latest", "content": content}]
        body = request_body(messages, tools=[tool_definition("read")])
        _, after = self.compress(encode_json(body), self.target)
        require(len(after["messages"]) == len(messages), "latest read turn lost a message")
        assert_tool_closure(after["messages"])
        require(after["messages"][2] == messages[2], "short read call changed")
        result = after["messages"][3]["content"]
        require(0 < len(result) < len(content) and is_subsequence(result, content), "large read result did not shrink by deletion")
        self.passed_case("one large read result shrinks within its retained tool turn")

    def long_write_arguments(self):
        content = "// source begins\n" + ''.join(
            f'const char * value_{index} = "中文 😀 \\\\ cache";\n' for index in range(256)
        ) + "// source ends\n"
        argument_values = {"path": "src/cache.cpp", "content": content, "mode": "write"}
        arguments = json.dumps(argument_values, ensure_ascii=True, indent=2)
        call = tool_call("write_latest", "write", arguments)
        messages = [SYSTEM, {"role": "user", "content": "Write the source file, then explain the change."},
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "write_latest", "content": "Wrote src/cache.cpp."}]
        body = request_body(messages, tools=[tool_definition("write")])
        prompt = self.client.json_request("/apply-template", body)["prompt"]
        require("value_255" in prompt, "the test template must include the large tool arguments")
        _, after = self.compress(encode_json(body), self.target)
        require(len(after["messages"]) == len(messages), "latest write turn lost a message")
        assert_tool_closure(after["messages"])
        result_arguments = after["messages"][2]["tool_calls"][0]["function"]["arguments"]
        expected_call = copy.deepcopy(call)
        expected_call["function"]["arguments"] = result_arguments
        require(after["messages"][2]["tool_calls"] == [expected_call], "write call metadata changed")
        require(after["messages"][3] == messages[3], "short write result changed")
        require(is_subsequence(result_arguments.encode("utf-8"), arguments.encode("utf-8")),
                "the inner arguments JSON was reserialized or rewritten")
        result_values = decode_json(result_arguments.encode("utf-8"))
        assert_json_string_deletions(argument_values, result_values)
        require(result_values["path"] == argument_values["path"] and result_values["mode"] == "write",
                "short write parameters changed")
        require(0 < len(result_values["content"]) < len(content), "large write content did not shrink")
        self.passed_case("large nested write content shrinks; JSON syntax, keys, and short values survive")

    def parallel_tool_turns(self):
        instruction = {"role": "user", "content": "Inspect these file pairs, then explain the latest result."}
        messages = [SYSTEM, instruction]
        for index in range(12):
            calls = [tool_call(f"read_{index}_{side}", "read", json.dumps({"path": f"{side}{index}.txt"}))
                     for side in ("a", "b")]
            messages += [
                {"role": "assistant", "content": None, "tool_calls": calls},
                {"role": "tool", "tool_call_id": calls[1]["id"], "content": "unchanged"},
                {"role": "tool", "tool_call_id": calls[0]["id"], "content": "unchanged"},
            ]
        body = request_body(messages, tools=[tool_definition("read")], parallel_tool_calls=True)
        original_count = self.client.prompt_tokens(body)
        _, after = self.compress(encode_json(body), min(self.target, original_count // 2), original_count)
        require(len(after["messages"]) < len(messages), "old tool exchanges under one user task were not removed")
        require(after["messages"][1] == instruction, "the current user task changed or disappeared")
        require(after["messages"][-3:] == messages[-3:], "the latest parallel tool exchange changed or disappeared")
        assert_tool_closure(after["messages"])
        original_turn = 0
        for message in after["messages"]:
            while original_turn < len(messages) and messages[original_turn] != message:
                original_turn += 1
            require(original_turn < len(messages), "short historical messages changed or moved")
            original_turn += 1
        self.passed_case("one user with many short tool exchanges compresses; the latest exchange stays closed")

    def long_reasoning_content(self):
        content = "Plan begins.\n" + "".join(
            f"Step {index}: inspect cached state and compare the unchanged value.\n" for index in range(256)
        ) + "analysis_done"
        call = tool_call("reasoning_latest", "read", '{"path":"src/cache.cpp"}')
        messages = [SYSTEM, {"role": "user", "content": "Review the cache behavior."},
                    {"role": "assistant", "content": "Ready.", "reasoning_content": content,
                     "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "reasoning_latest", "content": "unchanged"}]
        body = request_body(messages, tools=[tool_definition("read")],
                            chat_template_kwargs={"preserve_reasoning": True})
        prompt = self.client.json_request("/apply-template", body)["prompt"]
        require("analysis_done" in prompt, "the test template must include reasoning_content")
        _, after = self.compress(encode_json(body), self.target)
        require(len(after["messages"]) == len(messages), "the latest reasoning message disappeared")
        assert_tool_closure(after["messages"])
        result = after["messages"][-2]["reasoning_content"]
        expected = copy.deepcopy(messages[-2])
        expected["reasoning_content"] = result
        require(after["messages"][-2] == expected, "non-reasoning assistant fields changed")
        require(after["messages"][-1] == messages[-1], "the short reasoning tool result changed")
        require(0 < len(result) < len(content) and is_subsequence(result, content),
                "large reasoning_content did not shrink by deletion")
        self.passed_case("large reasoning_content in the current task can shrink")

    def strict_alternation(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "tool_calls": []},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        body = request_body(messages, chat_template_kwargs={"strict_alternation": True})
        invalid = copy.deepcopy(body)
        invalid["messages"] = messages[:1] + messages[2:]
        rejected = self.client.request("/apply-template", encode_json(invalid))
        require(rejected.status >= 400, "the strict test template must reject consecutive user messages")
        original_count = self.client.prompt_tokens(body)
        _, after = self.compress(encode_json(body), original_count - 1, original_count)
        result = after["messages"]
        require(0 < len(result) <= 3 and len(result) % 2 == 1,
                "strict alternation requires deleting at least one whole message pair")
        require(result[-1] == messages[-1], "the latest user disappeared under the strict template")
        for index, message in enumerate(result):
            require(message["role"] == ("user" if index % 2 == 0 else "assistant"),
                    "compressed messages violate strict alternation")
        self.passed_case("strict alternating template accepts the compressed request and keeps the latest user")

    def run(self):
        self.fit_and_stream()
        self.bad_inputs()
        self.impossible_budget()
        self.long_user_and_reuse()
        self.unicode_and_escapes()
        self.long_read_result()
        self.long_write_arguments()
        self.parallel_tool_turns()
        self.long_reasoning_content()
        self.strict_alternation()
        print(f"All {self.passed} compression HTTP cases passed.", flush=True)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args()
    require(args.ctx_size >= 1024, "use a context of at least 1024 tokens")
    port = free_port()
    client = Client(port, args.timeout)
    command = [str(args.binary.resolve()), "--model", str(args.model.resolve()),
               "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(args.ctx_size),
               "--batch-size", "256", "--ubatch-size", "64", "--parallel", "1",
               "--n-predict", "16", "--threads", "2", "--no-warmup", "--jinja",
               "--alias", "compression-test", "--kv-unified"]
    with tempfile.TemporaryDirectory(prefix="compress-http-", dir=Path.cwd()) as directory:
        template_path = Path(directory) / "chat.jinja"
        template_path.write_text(CHAT_TEMPLATE, encoding="utf-8")
        command += ["--chat-template-file", str(template_path)]
        log_path = Path(directory) / "server.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    require(process.poll() is None, "server exited during startup")
                    try:
                        if client.request("/health").status == 200:
                            break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    require(time.monotonic() < deadline, "server startup timed out")
                    time.sleep(0.1)
                Suite(client, args.ctx_size).run()
            except Exception:
                log.flush()
                print("\nServer log tail:\n" + log_path.read_text(errors="replace")[-12000:], flush=True)
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)


if __name__ == "__main__":
    main()
