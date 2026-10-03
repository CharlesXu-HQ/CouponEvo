import json
import unittest
from unittest.mock import patch

from promolift.provider import ApiProvider, request_json


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": '{"ok": true}'}}]}).encode()


class ProviderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
