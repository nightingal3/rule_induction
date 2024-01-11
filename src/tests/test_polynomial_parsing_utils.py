import pytest
from src.utils.polynomial_parsing_utils import parse_polynomial


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("y = x", {1: 1}),
        ("y = 5", {0: 5}),
        ("y = 3x + 4", {0: 4, 1: 3}),
        ("y = 6x + 4", {0: 4, 1: 6}),
        ("y = -17x + 5", {0: 5, 1: -17}),
        ("y = 3x - 4", {0: -4, 1: 3}),
        ("y = 3x^0 + 17x^1", {0: 3, 1: 17}),
        ("y = -15x^1 + 6x^0", {0: 6, 1: -15}),
        ("y = 50x^0 + 76x", {0: 50, 1: 76}),
        ("y = -14 + (-12)x", {0: -14, 1: -12}),
        ("y = -12.8x^0 + 21.6x^1", {0: -12.8, 1: 21.6}),
        ("y = -5 - (-6x)", {0: -5, 1: 6}),
        ("y = (8 - 315/19)x^0 + (-315/19)x^1", {}),
        ("Output = -12x^0 -14x^1", {0: -12, 1: -14}),
        ("y = ax + b", {}),
        ("y = ax^0 + bx^1", {}),
        ("y = 2x^2 + 3x^1 + 4x^0", {0: 4, 1: 3, 2: 2}),
    ],
    ids=[
        "no_coeff",
        "no_coeff_2",
        "simple-positive",
        "simple-positive-2",
        "simple-x1-negative",
        "simple-x0-negative",
        "simple-x0-x1",
        "simple-x0-x1-2",
        "simple-x0-x-mixed",
        "brackets",
        "floats",
        "double-negative",
        "fraction",
        "output",
        "declined-answer-1",
        "declined-answer-2",
        "quadratic",
    ],
)
def test_parse_polynomial(input_string, expected_output):
    assert parse_polynomial(input_string) == expected_output
