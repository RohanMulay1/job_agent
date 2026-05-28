"""
llm_brain.py — LLM intelligence layer.

Two public functions:
  evaluate_job_fit()          → bool   (cheap haiku model, runs per job card)
  parse_form_and_decide_actions() → FormResponse  (sonnet model, runs per form page)

Both use OpenRouter with the OpenAI-compatible client.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from config import settings
from profile_schema import CandidateProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter client (OpenAI-compatible)
# ---------------------------------------------------------------------------

_client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    default_headers={
        "HTTP-Referer": "https://github.com/RohanMulay1/job-agent",
        "X-Title": "Autonomous Job Agent",
    },
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class JobFitResult(BaseModel):
    match: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class FormAction(BaseModel):
    type: Literal["fill", "click", "select", "upload", "check", "skip"]
    selector: str
    value: str | None = None


class FormResponse(BaseModel):
    actions: list[FormAction]
    is_complete: bool = False
    needs_human: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JOB_FIT_SYSTEM = """\
You are a recruitment screening assistant. Given a job description and a candidate \
profile, decide whether the candidate is a reasonable match.

Rules:
- Score 0.0–1.0 where 1.0 = perfect match.
- Return match=true only if score >= {threshold}.
- Consider: required skills overlap, seniority alignment, domain fit (AI/ML/CV/LLM/ADAS).
- A student/intern with strong project experience CAN match entry/junior roles.
- Ignore salary, location, notice period — those are handled elsewhere.

Respond with ONLY valid JSON matching this schema (no markdown, no prose):
{{"match": bool, "score": float, "reason": "one sentence"}}
"""

_FORM_FILL_SYSTEM = """\
You are an autonomous job application assistant filling out an online application form \
on behalf of a candidate. You will receive:
1. A simplified HTML snapshot of the CURRENT form page (labels, inputs, selects, buttons).
2. The candidate's full profile as JSON.

Your task: return a JSON list of actions to complete THIS page of the form.

Action types:
- "fill"   → type text into an input/textarea  (value = text to type)
- "select" → choose a dropdown option          (value = visible option text)
- "click"  → click a button/radio/checkbox     (value = null)
- "check"  → tick a checkbox                   (value = null)
- "upload" → file upload field                 (value = absolute file path)
- "skip"   → intentionally leave blank         (value = reason why)

Selector rules:
- Prefer attribute selectors: input[name='x'], select[id='y'], textarea[name='z']
- For buttons use: button:has-text('Next'), input[type='submit']
- NEVER invent selectors not present in the HTML snapshot.

Special instructions:
- For "years of experience" fields, use the candidate's years_total_experience.
- For salary/CTC fields, use the candidate's desired_salary_min (in the job's currency).
- For "current CTC" fields where unknown, enter 0 or leave blank via "skip".
- For cover letter / additional info fields, write a 2-sentence pitch from the summary.
- If a question is ambiguous or risky (e.g. background check consent, legal agreements), \
  set needs_human=true and stop — do not guess.
- Set is_complete=true ONLY when the final submit button has been clicked.

Respond with ONLY valid JSON — no markdown, no explanation:
{{"actions": [...], "is_complete": bool, "needs_human": bool, "notes": "optional string"}}
"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


async def evaluate_job_fit(
    job_title: str,
    job_description: str,
    profile: CandidateProfile,
) -> JobFitResult:
    """
    Cheap haiku call — runs once per job card during the search phase.
    Returns a JobFitResult with match bool, score 0-1, and a short reason.
    """
    system_prompt = _JOB_FIT_SYSTEM.format(threshold=settings.fit_threshold)

    user_message = (
        f"JOB TITLE: {job_title}\n\n"
        f"JOB DESCRIPTION:\n{job_description[:3000]}\n\n"  # cap to keep tokens low
        f"CANDIDATE PROFILE:\n{profile.to_llm_summary()}"
    )

    logger.debug("evaluate_job_fit → model=%s title=%s", settings.screen_model, job_title)

    try:
        response = await _client.chat.completions.create(
            model=settings.screen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        # Strip markdown code fences if the model wraps the JSON anyway
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = JobFitResult.model_validate_json(raw)
        logger.info(
            "Job fit: %s | score=%.2f | match=%s | %s",
            job_title,
            result.score,
            result.match,
            result.reason,
        )
        return result

    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning("evaluate_job_fit parse error for '%s': %s", job_title, exc)
        # Fail open — let the agent attempt the application
        return JobFitResult(match=True, score=0.5, reason="LLM parse error — defaulting to match")

    except Exception as exc:
        logger.error("evaluate_job_fit API error for '%s': %s", job_title, exc)
        return JobFitResult(match=False, score=0.0, reason=f"API error: {exc}")


_COVER_LETTER_SYSTEM = """\
Write a 2-sentence cold outreach message for a startup job. Hard rules:

- EXACTLY 2 sentences. No more.
- Under 50 words total. Count every word.
- Sentence 1: one specific thing about what the company actually does (from the JD). No adjectives like "innovative", "impressive", "game-changer", "revolutionizing".
- Sentence 2: one concrete thing the candidate has built or done that is directly relevant. Be specific, not vague.
- NO: "I am writing", "I would be a great fit", "passionate", "leverage", "excited", "love to", "I believe", "looking forward", "would love to connect", "keen to", "eager".
- NO em dashes, NO hyphens as dashes, NO filler phrases.
- Do NOT mention YC or Y Combinator.
- Output ONLY the 2 sentences. Nothing else.
"""


async def generate_cover_letter(
    company: str,
    role: str,
    job_description: str,
    profile: CandidateProfile,
) -> str:
    """
    Generate a short, personalized YC-style application message.
    Uses the form model (70B) for quality.
    """
    top_skills = ", ".join(profile.expert_skills()[:6]) or profile.to_llm_summary()[:300]
    recent_exp = ""
    if profile.work_experience:
        exp = profile.work_experience[0]
        bullets = " | ".join(exp.description[:2])
        recent_exp = f"Most recent: {exp.title} at {exp.company} — {bullets}"

    user_message = (
        f"COMPANY: {company}\n"
        f"ROLE: {role}\n\n"
        f"JOB DESCRIPTION:\n{job_description[:3000]}\n\n"
        f"CANDIDATE:\n"
        f"Name: {profile.personal_info.full_name}\n"
        f"Summary: {profile.resume_summary}\n"
        f"Expert skills: {top_skills}\n"
        f"{recent_exp}"
    )

    logger.debug("generate_cover_letter → company=%s role=%s", company, role)

    try:
        response = await _client.chat.completions.create(
            model=settings.form_model,
            messages=[
                {"role": "system", "content": _COVER_LETTER_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=80,
        )
        text = (response.choices[0].message.content or "").strip()
        logger.info("Cover letter generated for %s @ %s (%d chars)", role, company, len(text))
        return text

    except Exception as exc:
        logger.error("generate_cover_letter error for %s @ %s: %s", role, company, exc)
        # Fallback: minimal honest message
        return (
            f"Hi, I'm {profile.personal_info.full_name}. "
            f"I came across the {role} role at {company} and wanted to reach out directly. "
            f"{profile.resume_summary} I'd love to learn more about what you're building."
        )


async def parse_form_and_decide_actions(
    dom_snapshot: str,
    profile: CandidateProfile,
    page_hint: str = "",
) -> FormResponse:
    """
    Sonnet call — runs once per form page/modal during the apply phase.
    Returns structured actions the agent should execute on the current DOM.

    dom_snapshot: simplified HTML (labels + inputs only, stripped of noise, <4 KB)
    page_hint:    optional context string, e.g. "Step 2 of 4: Work Experience"
    """
    profile_json = profile.model_dump_json(indent=2)

    user_message = (
        f"{f'FORM CONTEXT: {page_hint}' + chr(10) if page_hint else ''}"
        f"FORM HTML SNAPSHOT:\n{dom_snapshot[:4000]}\n\n"
        f"CANDIDATE PROFILE (JSON):\n{profile_json}"
    )

    logger.debug("parse_form → model=%s hint=%s", settings.form_model, page_hint)

    try:
        response = await _client.chat.completions.create(
            model=settings.form_model,
            messages=[
                {"role": "system", "content": _FORM_FILL_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = FormResponse.model_validate_json(raw)
        logger.info(
            "Form actions: %d actions | complete=%s | needs_human=%s | %s",
            len(result.actions),
            result.is_complete,
            result.needs_human,
            result.notes,
        )
        return result

    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning("parse_form parse error: %s", exc)
        # On parse failure, signal human takeover — don't guess blindly
        return FormResponse(
            actions=[],
            is_complete=False,
            needs_human=True,
            notes=f"LLM response parse error: {exc}",
        )

    except Exception as exc:
        logger.error("parse_form API error: %s", exc)
        return FormResponse(
            actions=[],
            is_complete=False,
            needs_human=True,
            notes=f"API error: {exc}",
        )
