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


@pytest.mark.parametrize(
    "idx, output_text, answer, expected_result",
    [
        (0, "Output: 5", 5, True),
        (1, "Output: 5", 10, False),
        (2, "Output: 100\n\nInput: 30\nOutput: 10", 100, True),
        (3, "Output: 100\n\nInput: 30\nOutput: 10", 10, False),
    ],
)
def test_math_answer_validation(
    function_task, idx, output_text, answer, expected_result
):
    result = function_task.validate_improved(idx, output_text, answer)
    assert result == expected_result
