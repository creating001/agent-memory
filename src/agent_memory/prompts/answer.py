from __future__ import annotations

from dataclasses import dataclass

from agent_memory.baseline.chunking import relative_time_annotation
from agent_memory.core.llm import extract_json_object
from agent_memory.core.schema import Example, RetrievedChunk
from agent_memory.prompts import answer_templates as templates


@dataclass(frozen=True)
class PromptProfile:
    """Named prompt bundle used by the memory QA pipeline."""

    name: str
    direct_answer: str
    general_requirements: str
    preference_requirements: str
    assistant_recall_requirements: str
    factual_extraction: str
    factual_answer: str
    aggregation_extraction: dict[str, str]
    aggregation_answer: dict[str, str]
    verification: str


_AGGREGATION_EXTRACTION_TEMPLATES = {
    "scoped_aggregation": templates.SCOPED_AGGREGATION_EXTRACTION_TEMPLATE,
    "set_operation": templates.SET_OPERATION_EXTRACTION_TEMPLATE,
}

_AGGREGATION_ANSWER_TEMPLATES = {
    "scoped_aggregation": templates.SCOPED_AGGREGATION_ANSWER_TEMPLATE,
    "set_operation": templates.SET_OPERATION_ANSWER_TEMPLATE,
}

_STABLE_PROFILE = PromptProfile(
    name="stable_memory_v1",
    direct_answer=templates.DIRECT_ANSWER_TEMPLATE,
    general_requirements=templates.GENERAL_REQUIREMENTS,
    preference_requirements=templates.PREFERENCE_REQUIREMENTS,
    assistant_recall_requirements=templates.ASSISTANT_RECALL_REQUIREMENTS,
    factual_extraction=templates.FACTUAL_EVIDENCE_EXTRACTION_TEMPLATE,
    factual_answer=templates.FACTUAL_EVIDENCE_ANSWER_TEMPLATE,
    aggregation_extraction=_AGGREGATION_EXTRACTION_TEMPLATES,
    aggregation_answer=_AGGREGATION_ANSWER_TEMPLATES,
    verification=templates.ANSWER_VERIFICATION_TEMPLATE,
)

_PROFILE_ALIASES = {
    "default": _STABLE_PROFILE,
    "stable": _STABLE_PROFILE,
    "stable_memory": _STABLE_PROFILE,
    "stable_memory_v1": _STABLE_PROFILE,
}


def prompt_profile_bundle(prompt_profile: str = "default") -> PromptProfile:
    key = str(prompt_profile or "default").strip().lower()
    return _PROFILE_ALIASES.get(key, _STABLE_PROFILE)


def answer_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    *,
    requirement_style: str = "general",
    max_context_chars: int = 0,
    context_relative_time_annotations: bool = False,
    detailed_answer: bool = False,
    prompt_profile: str = "default",
) -> list[dict[str, str]]:
    context = format_context(
        retrieved,
        max_chars=max_context_chars,
        relative_time_annotations=context_relative_time_annotations,
    )
    profile = prompt_profile_bundle(prompt_profile)
    prompt = profile.direct_answer.format(
        query=query_text(example),
        context_str=context,
        extra_requirements=answer_requirements(
            requirement_style,
            detailed_answer=detailed_answer,
            prompt_profile=prompt_profile,
        ),
    )
    return [{"role": "user", "content": prompt}]


def list_evidence_requirements(
    *,
    list_reasoning: bool,
) -> str:
    parts = []
    if list_reasoning:
        parts.append(templates.LIST_EVIDENCE_REQUIREMENTS)
    return "".join(parts)


def list_evidence_answer_requirements(
    *,
    list_reasoning: bool,
) -> str:
    parts = []
    if list_reasoning:
        parts.append(templates.LIST_EVIDENCE_ANSWER_REQUIREMENTS)
    return "".join(parts)


def evidence_table_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    *,
    max_context_chars: int = 0,
    temporal_reasoning: bool = False,
    duration_reasoning: bool = False,
    list_reasoning: bool = False,
    context_relative_time_annotations: bool = False,
    prompt_profile: str = "default",
) -> list[dict[str, str]]:
    template = evidence_table_template(prompt_profile=prompt_profile)
    extra_requirements = ""
    prompt = template.format(
        query=query_text(example),
        context_str=format_context(
            retrieved,
            max_chars=max_context_chars,
            relative_time_annotations=context_relative_time_annotations,
        ),
        aggregation_requirements=extra_requirements,
        list_requirements=list_evidence_requirements(
            list_reasoning=list_reasoning,
        ),
        temporal_requirements=temporal_and_duration_requirements(
            temporal_reasoning=temporal_reasoning,
            duration_reasoning=duration_reasoning,
        ),
    )
    return [{"role": "user", "content": prompt}]


def evidence_table_answer_messages(
    example: Example,
    evidence_json: str,
    *,
    temporal_reasoning: bool = False,
    duration_reasoning: bool = False,
    list_reasoning: bool = False,
    detailed_answer: bool = False,
    prompt_profile: str = "default",
) -> list[dict[str, str]]:
    template = evidence_table_answer_template(prompt_profile=prompt_profile)
    extra_requirements = ""
    prompt = template.format(
        query=query_text(example),
        evidence_json=evidence_json.strip(),
        aggregation_requirements=extra_requirements,
        list_requirements=list_evidence_answer_requirements(
            list_reasoning=list_reasoning,
        ),
        temporal_requirements=temporal_and_duration_requirements(
            temporal_reasoning=temporal_reasoning,
            duration_reasoning=duration_reasoning,
        ),
    )
    prompt = add_answer_detail_requirements(prompt, detailed_answer)
    return [{"role": "user", "content": prompt}]


def evidence_table_template(*, prompt_profile: str = "default") -> str:
    return prompt_profile_bundle(prompt_profile).factual_extraction.replace("{aggregation_requirements}\n", "")


def evidence_table_answer_template(*, prompt_profile: str = "default") -> str:
    return prompt_profile_bundle(prompt_profile).factual_answer.replace("{aggregation_requirements}\n", "")


def temporal_and_duration_requirements(
    *,
    temporal_reasoning: bool,
    duration_reasoning: bool,
) -> str:
    parts = []
    if temporal_reasoning:
        parts.append(templates.TEMPORAL_EXTRACTION_REQUIREMENTS)
    if duration_reasoning:
        parts.append(templates.DURATION_EXTRACTION_REQUIREMENTS)
    return "".join(parts)


def multi_evidence_table_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    *,
    max_context_chars: int = 0,
    mode: str = "default",
    context_relative_time_annotations: bool = False,
    prompt_profile: str = "default",
) -> list[dict[str, str]]:
    templates_by_mode = prompt_profile_bundle(prompt_profile).aggregation_extraction
    template = templates_by_mode.get(
        mode,
        templates_by_mode["scoped_aggregation"],
    )
    prompt = template.format(
        query=query_text(example),
        context_str=format_context(
            retrieved,
            max_chars=max_context_chars,
            relative_time_annotations=context_relative_time_annotations,
        ),
    )
    return [{"role": "user", "content": prompt}]


def multi_evidence_answer_messages(
    example: Example,
    evidence_json: str,
    *,
    mode: str = "default",
    detailed_answer: bool = False,
    prompt_profile: str = "default",
) -> list[dict[str, str]]:
    templates_by_mode = prompt_profile_bundle(prompt_profile).aggregation_answer
    template = templates_by_mode.get(
        mode,
        templates_by_mode["scoped_aggregation"],
    )
    prompt = template.format(query=query_text(example), evidence_json=evidence_json.strip())
    prompt = add_answer_detail_requirements(prompt, detailed_answer)
    return [{"role": "user", "content": prompt}]


def verify_answer_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    draft_answer: str,
    *,
    max_context_chars: int = 0,
    context_relative_time_annotations: bool = False,
) -> list[dict[str, str]]:
    prompt = prompt_profile_bundle().verification.format(
        query=query_text(example),
        context_str=format_context(
            retrieved,
            max_chars=max_context_chars,
            relative_time_annotations=context_relative_time_annotations,
        ),
        draft_answer=draft_answer,
    )
    return [{"role": "user", "content": prompt}]


def query_text(example: Example) -> str:
    if example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}"
    return example.question


def answer_requirements(
    style: str,
    *,
    detailed_answer: bool = False,
    prompt_profile: str = "default",
) -> str:
    if style == "assistant_recall":
        requirements = prompt_profile_bundle(prompt_profile).assistant_recall_requirements
    elif style == "preference":
        requirements = prompt_profile_bundle(prompt_profile).preference_requirements
    else:
        requirements = prompt_profile_bundle(prompt_profile).general_requirements
    if detailed_answer:
        requirements += templates.ANSWER_DETAIL_REQUIREMENTS
    return requirements


def add_answer_detail_requirements(prompt: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    marker = "\nOutput Format:\n"
    if marker in prompt:
        return prompt.replace(marker, f"{templates.ANSWER_DETAIL_REQUIREMENTS}{marker}", 1)
    return f"{prompt}{templates.ANSWER_DETAIL_REQUIREMENTS}"


def format_context(
    retrieved: list[RetrievedChunk],
    *,
    max_chars: int = 0,
    relative_time_annotations: bool = False,
) -> str:
    if not retrieved:
        return "None"
    blocks = []
    used = 0
    for chunk in retrieved:
        content = chunk.text
        if relative_time_annotations and "[Resolved relative time:" not in content:
            annotation = relative_time_annotation(content, chunk.date)
            if annotation:
                content = f"{content}\n[Resolved relative time: {annotation}]"
        block = "\n".join([f"### Memory {chunk.rank}", f"Date: {chunk.date}", "Content:", content])
        if max_chars > 0 and blocks and used + len(block) > max_chars:
            break
        if max_chars > 0 and not blocks and len(block) > max_chars:
            block = block[:max_chars].rstrip() + "\n...[truncated]"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def parse_answer(raw_response: str) -> str:
    value = extract_json_object(raw_response)
    if value and value.get("answer") is not None:
        return str(value["answer"]).strip()
    return raw_response.strip()
