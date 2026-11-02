# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run public-operator correctness under the generic launch policy."""

import pytest
from bias_gelu_paths import generic_execution

if __name__ == "__main__":
    with generic_execution():
        result = pytest.main(
            [
                "tests/megatron/test_bias_gelu_impl.py",
                "tests/megatron/test_bias_gelu_triton_regression.py",
                "-q",
                "-rs",
                "--tb=short",
            ]
        )
    raise SystemExit(result)
