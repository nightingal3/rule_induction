import pytest
from src import get_task
from src.prompt_openai import init_task
from types import SimpleNamespace


@pytest.fixture()
def function_task():
    fake_args = SimpleNamespace(
        dataset="functions",
        prompt_type="base",
        model="gpt-3.5-turbo",
        temp=0.0,
        num_few_shot_examples=5,
        split="simple",
        use_min_cover=True,
        output=None,
        start_ind=0,
        end_ind=10,
        degree=1,
    )
    task = init_task(fake_args)

    yield task


@pytest.fixture()
def function_task_cot_test():
    fake_args = SimpleNamespace(
        dataset="functions",
        prompt_type="zs-cot",
        model="gpt-3.5-turbo",
        temp=0.0,
        num_few_shot_examples=5,
        split="simple",
        use_min_cover=True,
        output=None,
        start_ind=0,
        end_ind=10,
        degree=1,
    )
    task = init_task(fake_args)

    yield task


@pytest.fixture()
def colours_task():
    fake_args = SimpleNamespace(
        dataset="colours",
        prompt_type="base",
        model="gpt-3.5-turbo",
        temp=0.0,
        num_few_shot_examples=5,
        split="simple",
        use_min_cover=True,
        output=None,
        start_ind=0,
        end_ind=10,
        prompt_in_loop=False,
    )

    task = init_task(fake_args)

    yield task


@pytest.fixture()
def colours_test_cot_test():
    fake_args = SimpleNamespace(
        dataset="colours",
        prompt_type="zs-cot",
        model="gpt-3.5-turbo",
        temp=0.0,
        num_few_shot_examples=5,
        split="simple",
        use_min_cover=True,
        output=None,
        start_ind=0,
        end_ind=10,
        prompt_in_loop=False,
    )

    task = init_task(fake_args)

    yield task


@pytest.mark.parametrize(
    "idx, output_text, answer, expected_result",
    [
        (0, "Output: 5", 5, True),
        (1, "Output: 5", 10, False),
        (2, "Output: 100\n\nInput: 30\nOutput: 10", 100, True),
        (3, "Output: 100\n\nInput: 30\nOutput: 10", 10, False),
        (4, "Output: -13", -13, True),
        (
            5,
            """Let's try to find a relationship between the input (x) and the output (y). We can start by looking at the differences between the outputs and see if there's a consistent operation that can be applied to the inputs to get those outputs.

For input -10, the output is -213. If we consider a polynomial function, we might start with a cubic function since the outputs seem to grow at a rate faster than a quadratic function would suggest.

Let's assume the function is of the form y = ax^3 + bx^2 + cx + d.

We can set up a system of equations using the given input-output pairs:

For input -10:
a(-10)^3 + b(-10)^2 + c(-10) + d = -213

For input 9:
a(9)^3 + b(9)^2 + c(9) + d = 167

For input 4:
a(4)^3 + b(4)^2 + c(4) + d = 67

For input -3:
a(-3)^3 + b(-3)^2 + c(-3) + d = -73

For input 20:
a(20)^3 + b(20)^2 + c(20) + d = 387

Now, let's solve this system of equations. However, since we don't have the means to solve this system algorithmically here, we'll have to look for a pattern in a different way.

Let's look at the differences between the inputs and outputs:

Input: -10 → Output: -213 (difference of 203)
Input: 9 → Output: 167 (difference of 158)
Input: 4 → Output: 67 (difference of 63)
Input: -3 → Output: -73 (difference of 70)
Input: 20 → Output: 387 (difference of 367)

The differences don't immediately suggest a simple arithmetic or geometric progression. However, if we look at the inputs and corresponding outputs, we might notice that the outputs are all odd numbers, and the magnitude of the outputs seems to increase as the magnitude of the inputs increases.

Let's try to find a pattern by examining the inputs and outputs more closely. We can look at the squares and cubes of the inputs:

-10^2 = 100, -10^3 = -1000
9^2 = 81, 9^3 = 729
4^2 = 16, 4^3 = 64
-3^2 = 9, -3^3 = -27
20^2 = 400, 20^3 = 8000

Now, let's see if we can relate these squares and cubes to the outputs:

For input -10, the output is -213, which is close to -10^3 + 10^2 - 23.
For input 9, the output is 167, which is close to 9^3 - 9^2 + 5.
For input 4, the output is 67, which is close to 4^3 + 4^2 + 3.
For input -3, the output is -73, which is close to -3^3 + 3^2 - 19.
For input 20, the output is 387, which is close to 20^3 - 20^2 - 13.

It seems that the function might be of the form y = x^3 - x^2 + (some constant).

To find the constant, let's look at one of the pairs and solve for it:

For input 4, output 67:
67 = 4^3 - 4^2 + constant
67 = 64 - 16 + constant
67 = 48 + constant
constant = 67 - 48
constant = 19

So, the function could be y = x^3 - x^2 + 19.

Let's test this with the input -6:

y = (-6)^3 - (-6)^2 + 19
y = -216 - 36 + 19
y = -252 + 19
y = -233

Therefore, the output for the input -6 would be -233.

Output: -233""",
            -233,
            True,
        ),
        (
            6,
            """Let's try to find a relationship between the input (x) and the output (y). We can start by looking at the differences between the outputs and see if there's a consistent operation that can be applied to the inputs to get those outputs.

For input -10, the output is -213. If we consider a polynomial function, we might start with a cubic function since the outputs seem to grow at a rate faster than a quadratic function would suggest.

Let's assume the function is of the form y = ax^3 + bx^2 + cx + d.

We can set up a system of equations using the given input-output pairs:

For input -10:
a(-10)^3 + b(-10)^2 + c(-10) + d = -213

For input 9:
a(9)^3 + b(9)^2 + c(9) + d = 167

For input 4:
a(4)^3 + b(4)^2 + c(4) + d = 67

For input -3:
a(-3)^3 + b(-3)^2 + c(-3) + d = -73

For input 20:
a(20)^3 + b(20)^2 + c(20) + d = 387

Now, let's solve this system of equations. However, since we don't have the means to solve this system algorithmically here, we'll have to look for a pattern in a different way.

Let's look at the differences between the inputs and outputs:

Input: -10 → Output: -213 (difference of 203)
Input: 9 → Output: 167 (difference of 158)
Input: 4 → Output: 67 (difference of 63)
Input: -3 → Output: -73 (difference of 70)
Input: 20 → Output: 387 (difference of 367)

The differences don't immediately suggest a simple arithmetic or geometric progression. However, if we look at the inputs and corresponding outputs, we might notice that the outputs are all odd numbers, and the magnitude of the outputs seems to increase as the magnitude of the inputs increases.

Let's try to find a pattern by examining the inputs and outputs more closely. We can look at the squares and cubes of the inputs:

-10^2 = 100, -10^3 = -1000
9^2 = 81, 9^3 = 729
4^2 = 16, 4^3 = 64
-3^2 = 9, -3^3 = -27
20^2 = 400, 20^3 = 8000

Now, let's see if we can relate these squares and cubes to the outputs:

For input -10, the output is -213, which is close to -10^3 + 10^2 - 23.
For input 9, the output is 167, which is close to 9^3 - 9^2 + 5.
For input 4, the output is 67, which is close to 4^3 + 4^2 + 3.
For input -3, the output is -73, which is close to -3^3 + 3^2 - 19.
For input 20, the output is 387, which is close to 20^3 - 20^2 - 13.

It seems that the function might be of the form y = x^3 - x^2 + (some constant).

To find the constant, let's look at one of the pairs and solve for it:

For input 4, output 67:
67 = 4^3 - 4^2 + constant
67 = 64 - 16 + constant
67 = 48 + constant
constant = 67 - 48
constant = 19

So, the function could be y = x^3 - x^2 + 19.

Let's test this with the input -6:

y = (-6)^3 - (-6)^2 + 19
y = -216 - 36 + 19
y = -252 + 19
y = -233

Therefore, the output for the input -6 would be -233.

Output: -233""",
            233,
            False,
        ),
    ],
)
def test_math_answer_validation(
    function_task, idx, output_text, answer, expected_result
):
    result = function_task.validate(idx, output_text, answer)
    assert result == expected_result


@pytest.mark.parametrize(
    "idx, output_text, answer, expected_result",
    [
        (0, "Output: red green", "red green", True),
        (1, "Output: red yellow", "red green", False),
        (
            2,
            "Output: red yellow\nInput: lug lug\nOutput: blue blue",
            "red yellow",
            True,
        ),
        (
            3,
            "Output: red yellow\nInput: lug lug\nOutput: blue blue",
            "blue blue",
            False,
        ),
        (
            4,
            """
    Output: red yellow

    Input: lug walm dax bluf""",
            "red yellow",
            True,
        ),
        (
            5,
            """First, let's break down the input into groups of two words. Here are the groups:
- wif zup
- dax zup

Now, let's look at the first group 'wif zup'. According to the provided input-output pairs, 'wif' corresponds to 'red' and 'zup' corresponds to 'yellow'. So the first group translates to 'red yellow'.

Next, let's look at the second group 'dax zup'. According to the provided input-output pairs, 'dax' corresponds to 'blue' and 'zup' corresponds to 'yellow'. So the second group translates to 'blue yellow'.

Finally, let's combine the translations of both groups to get the final answer. The translations are 'red yellow' and 'blue yellow'. So the overall translation is 'red yellow blue yellow'.

Therefore, the output preceded by 'Output:' is:
Output: red yellow blue yellow""",
            "red yellow green yellow",
            False,
        ),
        (
            6,
            """First, let's break down the input into groups of two words. Here are the groups:
- wif zup
- dax zup

Now, let's look at the first group 'wif zup'. According to the provided input-output pairs, 'wif' corresponds to 'red' and 'zup' corresponds to 'yellow'. So the first group translates to 'red yellow'.

Next, let's look at the second group 'dax zup'. According to the provided input-output pairs, 'dax' corresponds to 'blue' and 'zup' corresponds to 'yellow'. So the second group translates to 'blue yellow'.

Finally, let's combine the translations of both groups to get the final answer. The translations are 'red yellow' and 'blue yellow'. So the overall translation is 'red yellow blue yellow'.

Therefore, the output preceded by 'Output:' is:
Output: red yellow blue yellow""",
            "red yellow blue yellow",
            True,
        ),
        (
            7,
            """In the given input-output pairs, it seems like each input word is associated with a specific output color. Let's analyze the pairs to find any patterns or rules:

Input: lug dax
Output: blue green

Input: wif zup
Output: red yellow

Input: lug bluf
Output: blue blue

Input: wif walm
Output: red red red

From the given examples, we can observe that:
- Each input word consists of two unique letters.
- The first letter of the input word determines the color of the output.
- The second letter of the input word determines the number of times the color should be repeated in the output.

Based on this observation, let's try to solve the problem for the given input:

Input: zup lug dax zup

For the first word ""zup"":
- z -> red (as per the pattern)
- u -> since ""u"" is not present in any previous input word, we cannot determine the number of times the color should be repeated. So, we can assume it to be one time.

Therefore, the output for ""zup"" would be ""red"".

For the second word ""lug"":
- l -> blue (as per the pattern)
- u -> We cannot determine the number of times the color should be repeated because ""u"" was not present in any previous words. So, let's assume it is one time.
- g -> green (as per the pattern)

Therefore, the output for ""lug"" would be ""blue green"".

For the third word ""dax"":
- d -> blue (as per the pattern)
- a -> since ""a"" is not present in any previous input word, we cannot determine the number of times the color should be repeated. So, let's assume it is one time.
- x -> We cannot determine the color associated with ""x"" as it is not present in any previous input words. So, let's assume it to be a separate color.

Therefore, the output for ""dax"" would be ""blue separate_color"".

For the fourth word ""zup"":
- z -> red (as per the pattern)
- u -> since ""u"" was already present in the first word and it was associated with the color ""red"", we can assume the color to be repeated as ""red"".

Therefore, the output for ""zup"" would be ""red red"".

Putting it all together, the output for the given input ""zup lug dax zup"" would be: ""Output: red blue green blue separate_color red red"".

""",
            "yellow blue green yellow",
            False,
        ),
    ],
)
def test_colours_task_validation(
    colours_task, idx, output_text, answer, expected_result
):
    result = colours_task.validate(idx, output_text, answer)
    assert result == expected_result, f"Failed at index {idx}"


@pytest.mark.parametrize(
    "idx,output_text,answer,expected_result",
    [
        (0, "Final Output: red green", "red green", True),
        (1, "Final Output: red green", "red yellow", False),
        (
            2,
            """"Input: dax
Output: blue

Final Output: blue""",
            "blue",
            True,
        ),
        (
            3,
            """"Input: dax
Output: blue

Final Output: blue""",
            "green",
            False,
        ),
    ],
)
def test_colours_task_cot_validation(
    colours_test_cot_test, idx, output_text, answer, expected_result
):
    result = colours_test_cot_test.validate(idx, output_text, answer)
    assert result == expected_result, f"Failed at index {idx}"


@pytest.mark.parametrize(
    "idx, output_text, answer, expected_result",
    [
        (0, "Final Output: 5", 5, True),
        (1, "Final Output: 5", 10, False),
        (2, "Output: 100\n\nInput: 30\nOutput: 10\nFinal Output: 100", 100, True),
        (3, "Output: 100\n\nInput: 30\nOutput: 10\nFinal Output: 10", 10, True),
        (4, "Output: -13", -13, True),
        (
            5,
            """Output: -241

Based on the given input-output pairs, it seems that the function takes the input, multiplies it by 17, and then subtracts 8 from the result. So, the function can be represented as:

f(x) = 17x - 8

To verify this, let's apply the function to the given input:

f(-12) = 17(-12) - 8
       = -204 - 8
       = -212

Therefore, the final output is:

Final Output: -212""",
            -212,
            True,
        ),
    ],
)
def test_functions_task_cot_validation(
    function_task_cot_test, idx, output_text, answer, expected_result
):
    result = function_task_cot_test.validate(idx, output_text, answer)
    assert result == expected_result, f"Failed at index {idx}"
