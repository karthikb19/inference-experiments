"""Original five-shot MMLU prompt rendering."""

from inference_experiments.mmlu.models import CHOICES, MMLUExample


def render_example(example: MMLUExample, *, include_answer: bool) -> str:
    """Render one example in the original MMLU format."""
    lines = [example.question]
    lines.extend(
        f"{label}. {text}" for label, text in zip(CHOICES, example.choices, strict=True)
    )
    lines.append(f"Answer: {example.answer}" if include_answer else "Answer:")
    suffix = "\n\n" if include_answer else ""
    return "\n".join(lines) + suffix


def render_five_shot_prompt(
    example: MMLUExample,
    demonstrations: tuple[MMLUExample, ...],
) -> str:
    """Render the original MMLU header, five demonstrations, and test row."""
    if len(demonstrations) != 5:
        raise ValueError("five-shot prompts require exactly five demonstrations")
    if any(row.subject != example.subject for row in demonstrations):
        raise ValueError("demonstrations must match the test subject")
    subject = example.subject.replace("_", " ")
    header = (
        "The following are multiple choice questions (with answers) about "
        f"{subject}.\n\n"
    )
    few_shot = "".join(
        render_example(row, include_answer=True) for row in demonstrations
    )
    return header + few_shot + render_example(example, include_answer=False)
