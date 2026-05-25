from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field, HttpUrl, model_validator


class PersonalInfo(BaseModel):
    full_name: str
    email: str
    phone: str
    location: str
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: Optional[int] = None
    is_current: bool = False
    gpa: Optional[float] = Field(default=None, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def _check_years(self) -> Education:
        if self.end_year and self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")
        return self


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str  # "YYYY-MM" format
    end_date: Optional[str] = None  # None means current
    is_current: bool = False
    description: list[str] = Field(default_factory=list, description="Bullet-point achievements")
    technologies: list[str] = Field(default_factory=list)

    @property
    def years_experience(self) -> float:
        from datetime import datetime
        fmt = "%Y-%m"
        start = datetime.strptime(self.start_date, fmt)
        end = datetime.now() if self.is_current else datetime.strptime(self.end_date, fmt)
        return round((end - start).days / 365.25, 1)


class Skill(BaseModel):
    name: str
    category: Literal["technical", "soft", "tool", "language", "framework"]
    proficiency: Literal["beginner", "intermediate", "expert"] = "intermediate"


class CandidateProfile(BaseModel):
    personal_info: PersonalInfo
    resume_summary: str = Field(description="2-3 sentence professional summary")
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list, description="Spoken/written languages")

    # Application preferences
    years_total_experience: float = Field(ge=0)
    desired_salary_min: Optional[int] = None
    desired_salary_max: Optional[int] = None
    desired_salary_currency: str = "USD"
    willing_to_relocate: bool = False
    work_authorization: str = Field(
        description="e.g. 'US Citizen', 'OPT', 'H1B', 'Indian Citizen', 'No sponsorship needed'"
    )
    preferred_work_type: Literal["remote", "hybrid", "onsite", "any"] = "any"
    notice_period_days: int = Field(default=30, ge=0)

    def skills_by_category(self, category: str) -> list[str]:
        return [s.name for s in self.skills if s.category == category]

    def expert_skills(self) -> list[str]:
        return [s.name for s in self.skills if s.proficiency == "expert"]

    def to_llm_summary(self) -> str:
        """Compact text representation for LLM prompts."""
        lines = [
            f"Name: {self.personal_info.full_name}",
            f"Summary: {self.resume_summary}",
            f"Total experience: {self.years_total_experience} years",
            f"Work auth: {self.work_authorization}",
            f"Relocate: {self.willing_to_relocate}",
            f"Work type preference: {self.preferred_work_type}",
            f"Notice period: {self.notice_period_days} days",
            "",
            "SKILLS:",
        ]
        for skill in self.skills:
            lines.append(f"  - {skill.name} ({skill.category}, {skill.proficiency})")
        lines.append("")
        lines.append("EXPERIENCE:")
        for exp in self.work_experience:
            lines.append(f"  {exp.title} @ {exp.company} ({exp.start_date} – {'Present' if exp.is_current else exp.end_date})")
            for bullet in exp.description[:3]:
                lines.append(f"    • {bullet}")
        lines.append("")
        lines.append("EDUCATION:")
        for edu in self.education:
            lines.append(f"  {edu.degree} in {edu.field_of_study} — {edu.institution} ({edu.start_year}–{edu.end_year or 'Present'})")
        return "\n".join(lines)


def load_profile(path: Path | str) -> CandidateProfile:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CandidateProfile.model_validate(data)
