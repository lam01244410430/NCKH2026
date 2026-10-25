"""Chinese grammar explanation template."""
from ..schemas import ChatMessage, NonEmpty, PromptRecord, Record
from .common import task_messages


class IncorrectExample(Record):
    incorrect: NonEmpty
    correction: NonEmpty


class PracticeItem(Record):
    question: NonEmpty
    answer: NonEmpty


class GrammarOutput(Record):
    explanation: NonEmpty
    form: NonEmpty
    usage_constraints: list[NonEmpty]
    examples: list[NonEmpty]
    incorrect_example: IncorrectExample
    practice: PracticeItem


def render_grammar_prompt(record: PromptRecord) -> list[ChatMessage]:
    return task_messages(record, "grammar_explanation", (
        "结合主题及context解释grammar_point，考虑learner_l1的学习者背景。"
        "解释本身使用符合目标HSK等级的中文；不额外加入双语讲解。"
        "提供简明解释、形式和使用限制，恰好example_count个正确的语境例句，"
        "一个对比性的错误例句及改正，以及一个简短练习和答案。"
    ), GrammarOutput.model_json_schema())
