#!/usr/bin/env python3
from __future__ import annotations

import sys

import openpi
import transformers
from transformers.models.siglip import check


print(f"python={sys.executable}")
print(f"openpi={openpi.__file__}")
print(f"transformers={transformers.__file__}")
print(f"transformers_version={transformers.__version__}")
print(f"transformers_replace={check.check_whether_transformers_replace_is_installed_correctly()}")

assert openpi.__file__.startswith("/ssd/hhw/openpi-hzh/src/")
assert transformers.__version__ == "4.53.2"
assert check.check_whether_transformers_replace_is_installed_correctly()
