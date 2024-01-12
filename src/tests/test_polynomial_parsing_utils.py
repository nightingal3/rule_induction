import pytest
from src.utils.polynomial_parsing_utils import parse_polynomial


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("y = x", {1: 1}),
        ("y = 5", {0: 5}),
        ("y = -x - 1", {0: -1, 1: -1}),
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
        (
            """The given data cannot be accurately represented by a function in the form y = ax^0 + bx^1. However, by observing the pattern in the outputs, we can try to describe the relationship between inputs and outputs using a different form of function.

The difference between consecutive outputs appears to be decreasing by a constant amount, while the difference between consecutive inputs remains constant. This suggests that the relationship between inputs and outputs may be quadratic. Therefore, we can try to represent the data using a quadratic function in the form y = ax^2 + bx + c.

To find the coefficients a, b, and c, we can use the given data points and solve a system of linear equations. Let's denote the inputs as x and the outputs as y.

Using the data points:
(7, -98):  (-98) = a(7)^2 + b(7) + c
(17, -218): (-218) = a(17)^2 + b(17) + c
(5, -74):   (-74) = a(5)^2 + b(5) + c
(3, -50):   (-50) = a(3)^2 + b(3) + c
(-6, 58):   (58) = a(-6)^2 + b(-6) + c

Simplifying the above equations, we get:
49a + 7b + c = -98
289a + 17b + c = -218
25a + 5b + c = -74
9a + 3b + c = -50
36a - 6b + c = 58

Solving this system of equations, we can find the values of a, b, and c.""",
            {},
        ),
        (
            """Based on the provided inputs and outputs, it seems that the function can be described as a linear equation of the form:

Output: y = a + bx

To find the values of 'a' and 'b', we can use the given data points. Taking any two pairs of inputs and outputs, we can form a system of equations and solve for 'a' and 'b'.

From the given data points:

Input: 7, Output: -98
Input: 17, Output: -218

We can form the following equations:

-98 = a + 7b   --- (1)
-218 = a + 17b  --- (2)

Solving the system of equations gives us the values of 'a' and 'b'.

Subtracting equation (1) from equation (2):

-218 - (-98) = (a + 17b) - (a + 7b)
-218 + 98 = a + 17b - a - 7b
-120 = 10b
b = -12

Substituting the value of 'b' back into equation (1):

-98 = a + 7(-12)
-98 = a - 84
a = -14

Therefore, the function that captures the relationship between inputs and outputs is:

Output: y = -14 + (-12)x

Simplified form:
Output: y = -14 - 12x""",
            {0: -14, 1: -12},
        ),
        (
            """The given data points suggest a linear relationship between the inputs and outputs. To find the equation of the line, we can use linear regression to estimate the coefficients of the equation. 

Using the given data points, the equation of the line can be written as:

Output (y) = a * x^0 + b * x^1

To find the values of a and b, we can solve the following system of equations using the method of least squares:

1. For input -20 and output 265:
265 = a + (-20)b

2. For input -15 and output 195:
195 = a + (-15)b

3. For input -5 and output 55:
55 = a + (-5)b

4. For input -10 and output 125:
125 = a + (-10)b

5. For input 6 and output -99:
-99 = a + (6)b

Solving this system of equations, we can find the values of a and b.""",
            {},
        ),
        (
            """Based on the given inputs and outputs, we can observe that the relationship between the inputs and outputs is not linear. Let's try to find the best fit using a non-linear function.

To capture the relationship, we can use a quadratic function of the form:

Output: y = ax^2 + bx + c

Let's substitute the given inputs and outputs into the equation to solve for the values of a, b, and c:

Input: -20
Output: 265

265 = a(-20)^2 + b(-20) + c
265 = 400a - 20b + c

Input: -15
Output: 195

195 = a(-15)^2 + b(-15) + c
195 = 225a - 15b + c

Input: -5
Output: 55

55 = a(-5)^2 + b(-5) + c
55 = 25a - 5b + c

Input: -10
Output: 125

125 = a(-10)^2 + b(-10) + c
125 = 100a - 10b + c

Input: 6
Output: -99

-99 = a(6)^2 + b(6) + c
-99 = 36a + 6b + c

Now we have a system of linear equations which can be solved to determine the values of a, b, and c. Solving the system of equations, we find:

a = 0.5
b = -16.9
c = 69.8

Therefore, the function that captures the relationship between the inputs and outputs is:

Output: y = 0.5x^2 - 16.9x + 69.8""",
            {2: 0.5, 1: -16.9, 0: 69.8},
        ),
        (
            """The pattern in the given data suggests that the output is obtained by subtracting a value from the square of the input. Therefore, the function that captures this relationship can be written as:

Output = -15 - x^2

So, the function can be written as:

y = -15 - x^2""",
            {2: -1, 0: -15},
        ),
    ],
    ids=[
        "no_coeff",
        "no_coeff_2",
        "simple-negative-2",
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
        "long-no-answer",
        "long-with-answer",
        "long-no-answer-2",
        "long-quadratic",
        "long-quadratic-2",
    ],
)
def test_parse_polynomial(input_string, expected_output):
    assert parse_polynomial(input_string) == expected_output
