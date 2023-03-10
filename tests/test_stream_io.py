import unittest
from tensorizer import stream_io

class TestCurlStream(unittest.TestCase):
    def test_curl_stream(self):
        stream = stream_io.CURLStreamFile(
            "https://raw.githubusercontent.com/EleutherAI/gpt-neo/master/README.md")
        self.assertEqual(b"# GPT Neo", stream.read(9))

    def test_curl_stream_begin(self):
        stream = stream_io.CURLStreamFile(
            "https://raw.githubusercontent.com/EleutherAI/gpt-neo/master/README.md",
            begin=2)
        self.assertEqual(b"GPT Neo", stream.read(7))

    def test_curl_stream_end(self):
        stream = stream_io.CURLStreamFile(
            "https://raw.githubusercontent.com/EleutherAI/gpt-neo/master/README.md",
            end=2)
        self.assertEqual(b"# ", stream.read(5))

    def test_curl_stream_buffer_end(self):
        stream = stream_io.CURLStreamFile(
            "https://raw.githubusercontent.com/EleutherAI/gpt-neo/master/README.md",
            end=2)
        ba = bytearray(5)
        self.assertEqual(2, stream.readinto(ba))
        self.assertEqual(b"# \x00\x00\x00", ba)

    def test_curl_stream_seek(self):
        stream = stream_io.CURLStreamFile(
            "https://raw.githubusercontent.com/EleutherAI/gpt-neo/master/README.md")
        stream.seek(2)
        self.assertEqual(b"GPT Neo", stream.read(7))
        stream.seek(6)
        self.assertEqual(b"Neo\n", stream.read(4))

