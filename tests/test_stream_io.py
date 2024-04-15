import contextlib
import functools
import logging
import os
import re
import select
import subprocess
import tempfile
import threading
import typing
import unittest
from subprocess import Popen
from typing import List
from unittest.mock import patch

import boto3
import moto
import moto.server
import redis
import werkzeug.serving

from tensorizer import stream_io

# Logging of Boto3 ops in case you need it for moto confirmation.
# boto3.set_stream_logger(name='botocore')

NEO_URL = (
    "https://raw.githubusercontent.com/EleutherAI/gpt-neo/master/README.md"
)


class TestCurlStream(unittest.TestCase):
    def test_curl_stream(self):
        stream = stream_io.CURLStreamFile(NEO_URL)
        self.assertEqual(b"# GPT Neo", stream.read(9))

    def test_curl_stream_begin(self):
        stream = stream_io.CURLStreamFile(NEO_URL, begin=2)
        self.assertEqual(b"GPT Neo", stream.read(7))

    def test_curl_stream_end(self):
        stream = stream_io.CURLStreamFile(NEO_URL, end=2)
        self.assertEqual(b"# ", stream.read(5))

    def test_curl_stream_begin_and_end(self):
        stream = stream_io.CURLStreamFile(NEO_URL, begin=2, end=9)
        self.assertEqual(b"GPT Neo", stream.read())
        stream.close()

        stream = stream_io.CURLStreamFile(NEO_URL, begin=2, end=9)
        self.assertEqual(b"GPT Neo", stream.read(100))
        stream.close()

    def test_curl_stream_invalid_range(self):
        test = self.subTest

        def expect_err():
            return self.assertRaises(ValueError)

        with test("Invalid begin"), expect_err():
            stream_io.CURLStreamFile(NEO_URL, begin=-1)
        with test("Invalid end"), expect_err():
            stream_io.CURLStreamFile(NEO_URL, end=-1)
        with test("Invalid begin and end"), expect_err():
            stream_io.CURLStreamFile(NEO_URL, begin=-2, end=-1)
        with test("Backwards begin and end"), expect_err():
            stream_io.CURLStreamFile(NEO_URL, begin=10, end=5)
        with test("Equal begin and end"), expect_err():
            stream_io.CURLStreamFile(NEO_URL, begin=5, end=5)
        with test("End at zero (equals begin)"), expect_err():
            stream_io.CURLStreamFile(NEO_URL, end=0)

    def test_curl_stream_buffer_end(self):
        stream = stream_io.CURLStreamFile(NEO_URL, end=2)
        ba = bytearray(5)
        self.assertEqual(2, stream.readinto(ba))
        self.assertEqual(b"# \x00\x00\x00", ba)

    def test_curl_stream_seek(self):
        stream = stream_io.CURLStreamFile(NEO_URL)
        stream.seek(2)
        self.assertEqual(b"GPT Neo", stream.read(7))
        stream.seek(6)
        self.assertEqual(b"Neo\n", stream.read(4))

    def test_curl_stream_seek_forward(self):
        stream = stream_io.CURLStreamFile(NEO_URL)
        stream.seek(2)
        self.assertEqual(b"GPT Neo", stream.read(7))
        stream.seek(13)
        self.assertEqual(b"[DOI]", stream.read(5))


def set_up_moto(*endpoints):
    os.environ["MOTO_S3_CUSTOM_ENDPOINTS"] = ",".join(endpoints)

    # Clear any environment variables that boto3 may attempt to access
    # to avoid accidentally writing to a real bucket in case of failure
    old_environment = {
        key: os.environ[key]
        for key in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SECURITY_TOKEN",
            "AWS_SESSION_TOKEN",
            "AWS_DEFAULT_REGION",
        )
        if key in os.environ
    }

    for key in old_environment:
        test_value = "us-east-1" if key == "AWS_DEFAULT_REGION" else "TEST"
        os.environ[key] = test_value

    mock_s3 = moto.mock_s3()
    access_key = mock_s3.FAKE_KEYS["AWS_ACCESS_KEY_ID"]
    secret_key = mock_s3.FAKE_KEYS["AWS_SECRET_ACCESS_KEY"]
    return mock_s3, access_key, secret_key, old_environment


def tear_down_moto(old_environment):
    for key, value in old_environment:
        os.environ[key] = value


def _moto_app():
    return moto.server.DomainDispatcherApplication(
        moto.server.create_backend_app
    )


@contextlib.contextmanager
def mock_certs(cn="localhost"):
    with tempfile.TemporaryDirectory(suffix="_tensorizer_test_certs") as path:
        yield werkzeug.serving.make_ssl_devcert(
            base_path=os.path.join(path, "cert"), cn=cn
        )


@contextlib.contextmanager
def mock_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    ssl_context: typing.Union[
        None, typing.Tuple[str, str], typing.Literal["adhoc"]
    ] = None,
) -> str:
    # Disable mock server logs
    werkzeug_logger = logging.getLogger("werkzeug")
    old_log_level = werkzeug_logger.getEffectiveLevel()
    werkzeug_logger.setLevel(logging.CRITICAL + 1)

    server = werkzeug.serving.make_server(
        host=host,
        port=port,
        app=_moto_app(),
        threaded=True,
        ssl_context=ssl_context,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Disable https validation on endpoints
    ensure_https_endpoint, stream_io._ensure_https_endpoint = (
        stream_io._ensure_https_endpoint,
        lambda endpoint: endpoint,
    )
    try:
        scheme = "https" if ssl_context is not None else "http"
        yield f"{scheme}://{host}:{port:d}"
    finally:
        stream_io._ensure_https_endpoint = ensure_https_endpoint
        server.shutdown()
        server_thread.join(timeout=30)
        werkzeug_logger.setLevel(old_log_level)


def start_redis(port=6381) -> (Popen, redis.Redis):
    # Start redis server
    redis_process = Popen(
        [
            "redis-server",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
    )
    # Read output until ready
    output = b""
    time_limit = 5
    while b"Ready to accept connections" not in output and time_limit > 0:
        poll_result = select.select([redis_process.stdout], [], [], time_limit)[
            0
        ]
        if poll_result:
            line = redis_process.stdout.readline()
            output += line
            time_limit -= 1
        else:
            # Log the output in case of failure
            print(output)
            raise Exception("Redis server did not start")
    redis_client = redis.Redis(host="localhost", port=port, db=0)
    return redis_process, redis_client


def teardown_redis(redis_process: Popen, redis_client: redis.Redis):
    redis_client.flushall()
    redis_client.close()
    redis_process.stdout.close()
    redis_process.kill()
    redis_process.wait()


class TestRedis(unittest.TestCase):
    redis_uri: str
    redis_host: str
    redis_port: int
    hello_str: bytes
    world_str: bytes
    ex_str: bytes
    hello_k: str
    world_k: str
    ex_k: str
    keys: List[str]
    redis: Popen
    redis_client: redis.Redis

    @classmethod
    def setUpClass(cls):
        cls.redis_uri = "redis://localhost:6381"
        cls.redis_host = cls.redis_uri.split("://")[1].split(":")[0]
        cls.redis_port = int(cls.redis_uri.split("://")[1].split(":")[1])

        cls.hello_str = b"Hello" * 1024 * 1024
        cls.world_str = b"World" * 64 * 1024
        cls.ex_str = b"!" * 1024 * 1024
        cls.hello_k = "t:hello:0"
        cls.world_k = f"t:world:{len(cls.hello_str)}"
        cls.ex_k = f"t:excl:{len(cls.hello_str) + len(cls.world_str)}"

    @classmethod
    def tearDown(cls):
        teardown_redis(cls.redis, cls.redis_client)

    @classmethod
    def setUp(cls):
        # Start redis server
        cls.redis, cls.redis_client = start_redis(cls.redis_port)

        # Populate redis with data
        cls.redis_client.set(cls.hello_k, cls.hello_str)
        cls.redis_client.set(cls.world_k, cls.world_str)
        cls.redis_client.set(cls.ex_k, cls.ex_str)

    def test_redis_stream(self):
        with stream_io.open_stream(
            f"redis://{self.redis_host}:{self.redis_port}/t",
            mode="rb",
        ) as s:
            with self.subTest("test basic 1024 byte read"):
                self.assertEqual(s.read(1024), self.hello_str[:1024])

            with self.subTest("test read that spans entire key"):
                s.seek(0)
                self.assertEqual(s.read(len(self.hello_str)), self.hello_str)

            with self.subTest("test seeking and read that begins offset"):
                s.seek(len(self.hello_str) - 16)
                self.assertEqual(s.read(16), self.hello_str[-16:])

            with self.subTest(
                "test seeking and read that spans two key subsets"
            ):
                s.seek(len(self.hello_str) - 16)
                overlap = self.hello_str[-16:] + self.world_str[:1008]
                readOverlap = s.read(1024)
                self.assertEqual(readOverlap, overlap)

            with self.subTest("test reading from all keys"):
                s.seek(0)
                self.assertEqual(
                    s.read(
                        len(self.hello_str)
                        + len(self.world_str)
                        + len(self.ex_str)
                    ),
                    self.hello_str + self.world_str + self.ex_str,
                )
            # Report on our latencies
            setup_latency_ms = s.setup_latency * 1000
            response_latencies = ", ".join(
                [f"{r * 1000:0.2f}ms" for r in s.response_latencies]
            )
            print(f"Redis setup latency: {setup_latency_ms:0.2f}ms")
            print(f"Redis response latencies: {response_latencies}")


class TestS3(unittest.TestCase):
    endpoint: str
    region: str
    BUCKET_NAME: str
    ACCESS_KEY: str
    SECRET_KEY: str
    old_environment: dict

    @classmethod
    def setUpClass(cls):
        # Sets up a mock S3 environment with moto.
        # Can be replaced for testing without mocks.
        cls.endpoint = "https://" + stream_io.default_s3_write_endpoint
        cls.region = "ord1"  # must match the region for the endpoint above
        cls.BUCKET_NAME = "test-bucket"
        (
            cls.mock_s3,
            cls.ACCESS_KEY,
            cls.SECRET_KEY,
            cls.old_environment,
        ) = set_up_moto(cls.endpoint)

    @classmethod
    def tearDownClass(cls) -> None:
        tear_down_moto(cls.old_environment)

    def setUp(self) -> None:
        # This is in setUp/tearDown rather than setUpClass/tearDownClass
        # so that mock bucket state is not shared between test runs
        self.mock_s3.start()
        s3 = boto3.resource("s3", endpoint_url=self.endpoint)
        bucket = s3.Bucket(self.BUCKET_NAME)
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/bucket/create.html
        bucket.create(
            CreateBucketConfiguration={
                # This must match the test endpoints name
                "LocationConstraint": self.region,
            },
        )

    def tearDown(self) -> None:
        self.mock_s3.stop()

    def assert_bucket_contents(self, key, content):
        # Not a test case
        s3 = boto3.resource("s3")
        obj = s3.Object(self.BUCKET_NAME, key)
        actual = obj.get()["Body"].read()
        self.assertEqual(actual, content)

    def put_bucket_contents(self, key, content):
        # Not a test case
        s3 = boto3.resource("s3")
        obj = s3.Object(self.BUCKET_NAME, key)
        obj.put(Body=content)

    @patch.object(stream_io, "_s3_default_config_paths", ())
    def test_upload(self):
        key = "model.tensors"
        s = stream_io.open_stream(
            f"s3://{self.BUCKET_NAME}/{key}",
            mode="wb",
            s3_access_key_id=self.ACCESS_KEY,
            s3_secret_access_key=self.SECRET_KEY,
            s3_endpoint=self.endpoint,
        )
        long_string = b"Hello" * 1024
        s.write(long_string)
        s.close()
        self.assert_bucket_contents(key, long_string)

    @patch.object(stream_io, "_s3_default_config_paths", ())
    def test_download_url(self):
        # s3_download_url does interesting things with passing a config, so
        # needs specific test handling.
        with mock_server() as endpoint:
            key = "model.tensors"
            url = stream_io._s3_download_url(
                path_uri=f"s3://{self.BUCKET_NAME}/{key}",
                s3_access_key_id="X",
                s3_secret_access_key="X",
                s3_endpoint=endpoint,
            )
            # http://127.0.0.1:5000/test-bucket/model.tensors?AWSAccessKeyId=x&Signature=x&Expires=1690783200
            # http://127.0.0.1:5000/test-bucket/model.tensors?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=X%2F20230803%2Ford1%2Fs3%2Faws4_request&X-Amz-Date=20230803T204528Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=x

            # TODO: add test for s3_access_key_id="" => unsigned
            self.assertRegex(url, re.escape(f"{self.BUCKET_NAME}/{key}?"))
            V4_regex = [
                r"X-Amz-Expires=[0-9]+",
                r"X-Amz-Credential=[A-Z]+",
            ]
            V2_regex = [
                r"Expires=1[0-9]+",
                r"AWSAccessKeyId=[A-Z]+",
            ]

            # This should be tweaked to say if it's meant to be v4 or v2
            if "X-Amz-Algorithm=" in url:
                regex_present = V4_regex
                regex_absent = V2_regex
            elif "AWSAccessKeyId=" in url:
                regex_present = V2_regex
                regex_absent = V4_regex
            else:
                regex_present = []
                regex_absent = V4_regex + V2_regex

            for r in regex_present:
                self.assertRegex(url, r)
            for r in regex_absent:
                self.assertNotRegex(url, r)

    @patch.object(stream_io, "_s3_default_config_paths", ())
    def test_download(self):
        # CURLStreamFiles ignore moto's normal mocks, since moto still
        # generates a real URL when a real endpoint is used, so instead
        # we create an entire mock S3 server.
        with mock_server() as endpoint:
            key = "model.tensors"
            long_string = b"Hello" * 1024
            self.put_bucket_contents(key, long_string)
            self.assert_bucket_contents(key, long_string)
            with stream_io.open_stream(
                f"s3://{self.BUCKET_NAME}/{key}",
                mode="rb",
                s3_access_key_id="X",
                s3_secret_access_key="X",
                s3_endpoint=endpoint,
            ) as s:
                self.assertEqual(s.read(), long_string)

    @patch.object(stream_io, "_s3_default_config_paths", ())
    def test_download_certificate_handling(self):
        long_string = b"Hello" * 1024
        key = "model.tensors"

        def _init_bucket():
            self.put_bucket_contents(key, long_string)
            self.assert_bucket_contents(key, long_string)

        stream_opener = functools.partial(
            stream_io.open_stream,
            path_uri=f"s3://{self.BUCKET_NAME}/{key}",
            mode="rb",
            s3_access_key_id="X",
            s3_secret_access_key="X",
        )

        rejected_cert_regex = "Failed to open stream|cURL exit code 60"
        allow_untrusted = stream_io.CAInfo(allow_untrusted=True)

        with mock_server(ssl_context="adhoc") as endpoint:
            _init_bucket()

            with self.subTest(
                msg="Testing suppressed certificate verification"
            ):
                with stream_opener(
                    s3_endpoint=endpoint, certificate_handling=allow_untrusted
                ) as s:
                    self.assertEqual(s.read(), long_string)

            with self.subTest(msg="Testing normal certificate verification"):
                with self.assertRaisesRegex(
                    IOError, rejected_cert_regex
                ), stream_opener(
                    s3_endpoint=endpoint,
                    certificate_handling=None,
                ):
                    pass

        with mock_certs() as (cert_file, key_file):
            custom_cacert = stream_io.CAInfo(cacert=cert_file)

            with mock_server(
                host="localhost", ssl_context=(cert_file, key_file)
            ) as endpoint:
                _init_bucket()

                with self.subTest(
                    msg="Testing custom CA cert with custom handling"
                ):
                    with stream_opener(
                        s3_endpoint=endpoint, certificate_handling=custom_cacert
                    ) as s:
                        self.assertEqual(s.read(), long_string)

                with self.subTest(
                    msg="Testing custom CA cert with normal handling"
                ):
                    with self.assertRaisesRegex(
                        IOError, rejected_cert_regex
                    ), stream_opener(
                        s3_endpoint=endpoint,
                        certificate_handling=None,
                    ):
                        pass

        with mock_server(ssl_context=None) as endpoint:
            # This test checks that custom certificate handling has no effect on
            # regular HTTP operation, which doesn't involve certificates
            _init_bucket()

            with self.subTest(
                msg="Testing suppressed certificate verification with HTTP"
            ):
                with stream_opener(
                    s3_endpoint=endpoint, certificate_handling=allow_untrusted
                ) as s:
                    self.assertEqual(s.read(), long_string)

            with self.subTest(msg="Testing custom CA cert with HTTP"):
                with stream_opener(
                    s3_endpoint=endpoint, certificate_handling=custom_cacert
                ) as s:
                    self.assertEqual(s.read(), long_string)
