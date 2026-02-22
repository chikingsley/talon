from __future__ import absolute_import
from unittest.mock import *

import talon

# Pytest replacements for nose.tools
def eq_(a, b, msg=None):
    assert a == b, msg

def ok_(x, msg=None):
    assert x, msg

def assert_true(x, msg=None):
    assert x, msg

def assert_false(x, msg=None):
    assert not x, msg

def assert_in(member, container, msg=None):
    assert member in container, msg

def assert_not_in(member, container, msg=None):
    assert member not in container, msg

def assert_raises(exception, callable_obj, *args, **kwargs):
    try:
        callable_obj(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"{exception} not raised")


EML_MSG_FILENAME = "tests/fixtures/standard_replies/yahoo.eml"
MSG_FILENAME_WITH_BODY_SUFFIX = ("tests/fixtures/signature/emails/P/"
                                 "johndoeexamplecom_body")
EMAILS_DIR = "tests/fixtures/signature/emails"
TMP_DIR = "tests/fixtures/signature/tmp"

STRIPPED = "tests/fixtures/signature/emails/stripped/"
UNICODE_MSG = ("tests/fixtures/signature/emails/P/"
               "unicode_msg")


talon.init()
