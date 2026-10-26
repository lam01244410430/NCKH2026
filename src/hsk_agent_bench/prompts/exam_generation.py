"""Chinese multiple-choice template and structural output validation."""
from typing import Annotated

from pydantic import Field

from ..schemas import ChatMessage, NonEmpty, PromptRecord, Record
from .common import task_messages


class ExamItem(Record):
    question: NonEmpty
    options: Annotated[list[NonEmpty], Field(min_length=4, max_length=4)]
    answer: NonEmpty
    rationale: NonEmpty
    target_vocabulary: list[NonEmpty]
    target_grammar: list[NonEmpty]


class ExamOutput(Record):
    source_passage: NonEmpty
    items: Annotated[list[ExamItem], Field(min_length=1)]


def render_exam_prompt(record: PromptRecord) -> list[ChatMessage]:
    return task_messages(record, "exam_generation", (
        "编写source_length汉字左右的主题阅读材料，并按item_type生成恰好num_items道题。"
        "每题包含问题、四个不同的选项、唯一正确答案、简短答案依据、目标词汇和目标语法。"
        "options使用四个完整选项文本，answer必须逐字等于其中唯一一个选项文本。"
        "题目须能根据材料作答，干扰项合理但不能也是正确答案。rationale只写答案依据，"
        "不写推理过程；答案依据保留给验证人员，不展示给考生。"
    ), ExamOutput.model_json_schema())


def validate_exam_payload(payload: ExamOutput) -> None:
    """Reject structural ambiguity; semantic answerability needs expert review."""
    payload = ExamOutput.model_validate(payload.model_dump())
    for item in payload.items:
        options = [option.strip() for option in item.options]
        if len(set(options)) != 4:
            raise ValueError("duplicated options or multiple matching answers")
        if options.count(item.answer.strip()) != 1:
            raise ValueError("answer must match exactly one option")
