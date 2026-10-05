import json
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

from couponevo.provider import ApiProvider, request_json


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": '{"ok": true}'}}]}).encode()


class ProviderTests(unittest.TestCase):
    def test_stage_budgets_and_timeout_configuration(self):
        provider = ApiProvider("https://example.test", "model", "key",
                               request_timeout_seconds=600,
                               token_budgets={"proposal": 65536, "reflection": 20000})
        self.assertEqual(provider.token_budget("proposal", 10000), 65536)
        self.assertEqual(provider.token_budget("analysis", 10000), 10000)
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as call:
            request_json(provider, "high", [], max_tokens=100)
            self.assertEqual(call.call_args.kwargs["timeout"], 600)
            request_json(provider, "high", [], max_tokens=100, timeout=7.5)
            self.assertEqual(call.call_args.kwargs["timeout"], 7.5)

    def test_unconfigured_timeout_uses_legacy_default(self):
        provider = ApiProvider("https://example.test", "model", "key")
        self.assertIsNone(provider.request_timeout_seconds)
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as call:
            request_json(provider, "high", [], max_tokens=100)
        self.assertEqual(call.call_args.kwargs["timeout"], 180)

    def test_invalid_timeout_and_stage_budgets_are_rejected(self):
        for value in (0, -1, True, "600", float("nan"), float("inf")):
            with self.subTest(timeout=value), self.assertRaisesRegex(ValueError, "request_timeout_seconds"):
                ApiProvider("https://example.test", "model", "key", request_timeout_seconds=value)
        for value in (None, [], {"unknown": 100}, {"analysis": 0}, {"review": -1},
                      {"reflection": True}, {"proposal": 3.5}, {"diagnosis": "100"}):
            with self.subTest(budgets=value), self.assertRaisesRegex(ValueError, "token_budgets"):
                ApiProvider("https://example.test", "model", "key", token_budgets=value)

    def test_invalid_final_json_preserves_content_without_exposing_it_in_error(self):
        content = '{"private_marker":'

        class InvalidResponse(FakeResponse):
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": content, "reasoning_content": "private reasoning"}}]}).encode()

        with patch("urllib.request.urlopen", return_value=InvalidResponse()):
            with self.assertRaises(ValueError) as caught:
                request_json(ApiProvider("https://example.test", "model", "key"), "high", [], max_tokens=100)
        self.assertEqual(type(caught.exception).__name__, "InvalidJSONResponseError")
        self.assertEqual(caught.exception.content, content)
        self.assertIsInstance(caught.exception.parse_error, json.JSONDecodeError)
        self.assertNotIn("private_marker", str(caught.exception))
        self.assertNotIn("private_marker", repr(caught.exception))
        self.assertNotIn("private reasoning", repr(caught.exception))

    def test_base_url_and_key_drive_high_request_with_thinking_enabled(self):
        provider = ApiProvider("https://api.deepseek.com", "deepseek-flash", "test-key",
                               thinking="enabled")
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as call:
            self.assertEqual(request_json(provider, "high", [{"role": "user", "content": "JSON please"}],
                                          max_tokens=100), {"ok": True})
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("test-key", repr(provider))

    def test_full_endpoint_omits_vendor_thinking_but_keeps_max_effort(self):
        provider = ApiProvider("https://example.com/v1/chat/completions", "model-a", "test-key",
                               thinking="omit")
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as call:
            request_json(provider, "max", [{"role": "user", "content": "JSON please"}], max_tokens=100)
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions")
        self.assertNotIn("thinking", body)
        self.assertEqual(body["reasoning_effort"], "max")

    def test_disabled_thinking_cannot_silently_nullify_high_effort(self):
        with self.assertRaisesRegex(ValueError, "thinking"):
            ApiProvider("https://api.deepseek.com", "deepseek-flash", "test-key",
                        thinking="disabled")

    def test_non_object_model_response_is_rejected(self):
        class ListResponse(FakeResponse):
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": "[]"}}]}).encode()

        provider = ApiProvider("https://api.deepseek.com", "deepseek-flash", "test-key")
        with patch("urllib.request.urlopen", return_value=ListResponse()):
            with self.assertRaisesRegex(ValueError, "JSON object"):
                request_json(provider, "high", [], max_tokens=100)

    def test_incomplete_response_exposes_finish_reason(self):
        class LengthResponse(FakeResponse):
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "length", "message": {
                    "content": '{"partial":'}}]}).encode()

        provider = ApiProvider("https://example.test", "model", "key")
        with patch("urllib.request.urlopen", return_value=LengthResponse()):
            with self.assertRaises(RuntimeError) as captured:
                request_json(provider, "high", [{"role": "user", "content": "test"}], max_tokens=100)
        self.assertEqual(type(captured.exception).__name__, "IncompleteResponseError")
        self.assertEqual(getattr(captured.exception, "reason", None), "length")
        self.assertEqual(captured.exception.content, '{"partial":')
        self.assertNotIn("partial", str(captured.exception))

    def test_keepalive_chunks_cannot_extend_total_deadline_and_reader_closes(self):
        class KeepaliveResponse(FakeResponse):
            def __init__(self):
                self.closed = threading.Event()
                self.reads = 0

            def read(self):
                time.sleep(0.4)
                return super().read()

            def read1(self, _size):
                self.reads += 1
                time.sleep(0.01)
                return b" \n"

            def __exit__(self, *_):
                self.closed.set()
                return False

        response = KeepaliveResponse()
        provider = ApiProvider("https://example.test", "model", "key")
        with patch("urllib.request.urlopen", return_value=response):
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                request_json(provider, "high", [], max_tokens=100, timeout=0.1)
            elapsed = time.monotonic() - start
            self.assertTrue(response.closed.wait(0.4))
        self.assertLess(elapsed, 0.3)
        self.assertGreater(response.reads, 0)

    def test_chunked_normal_response_is_decoded(self):
        class ChunkedResponse(FakeResponse):
            def __init__(self):
                self.parts = iter((b'{"choices":[{"finish_reason":"stop",',
                                   b'"message":{"content":"{\\"ok\\": true}"}}]}', b""))

            def read(self):
                raise AssertionError("HTTPResponse.read1 should be used")

            def read1(self, _size):
                return next(self.parts)

        provider = ApiProvider("https://example.test", "model", "key")
        with patch("urllib.request.urlopen", return_value=ChunkedResponse()):
            self.assertEqual(request_json(provider, "high", [], max_tokens=100), {"ok": True})

    def test_connection_delay_is_inside_total_deadline(self):
        class ClosingResponse(FakeResponse):
            def __init__(self):
                self.closed = threading.Event()

            def __exit__(self, *_):
                self.closed.set()
                return False

        response = ClosingResponse()

        def delayed_open(*_args, **_kwargs):
            time.sleep(0.4)
            return response

        provider = ApiProvider("https://example.test", "model", "key")
        with patch("urllib.request.urlopen", side_effect=delayed_open):
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                request_json(provider, "high", [], max_tokens=100, timeout=0.1)
            elapsed = time.monotonic() - start
            self.assertTrue(response.closed.wait(0.5))
        self.assertLess(elapsed, 0.3)

    def test_http_error_still_reports_status_without_credentials(self):
        provider = ApiProvider("https://example.test", "model", "secret-key")
        error = urllib.error.HTTPError("https://example.test", 429, "Too Many Requests", {}, None)
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as captured:
                request_json(provider, "high", [], max_tokens=100, timeout=0.05)
        self.assertIn("HTTP 429", str(captured.exception))
        self.assertNotIn("secret-key", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
