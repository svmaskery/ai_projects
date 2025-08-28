from pydantic import BaseModel, Field, conlist
from typing import List


class Experience(BaseModel):
    """Represents a single work experience entry on a resume."""
    job_title: str = Field(..., description="The user's job title for this experience.")
    company: str = Field(..., description="The name of the company.")
    years: str = Field(..., description="The years of employment (e.g., '2018-2022').")
    bullet_points: List[str] = Field(..., description="A list of bullet points describing responsibilities and achievements.")

class ResumeModel(BaseModel):
    """The structured schema for a user's resume."""
    name: str = Field(..., description="The user's full name.")
    contact: str = Field(..., description="The user's contact information.")
    summary: str = Field(..., description="The user's professional summary.")
    experience: List[Experience] = Field(..., description="A list of work experiences.")
    skills: List[str] = Field(..., description="A list of the user's skills.")
    education: str = Field(..., description="The user's educational background.")

class JobDescriptionModel(BaseModel):
    """The structured schema for a job description."""
    job_title: str = Field(..., description="The title of the job.")
    company: str = Field(..., description="The name of the company hiring.")
    responsibilities: List[str] = Field(..., description="A list of key responsibilities for the role.")
    required_skills: List[str] = Field(..., description="A list of required technical and soft skills.")

class RewrittenBulletPoint(BaseModel):
    """A suggestion for rewriting a single bullet point."""
    original: str = Field(..., description="The original bullet point from the resume.")
    rewritten: str = Field(..., description="The rewritten, more impactful version of the bullet point.")
    reasoning: str = Field(..., description="A brief explanation for why the rewrite is better.")

class FeedbackModel(BaseModel):
    """The final structured feedback provided to the user."""
    overall_score: float = Field(..., ge=0, le=100, description="An overall match score (0-100) for the resume against the job description.")
    missing_skills: conlist(str, min_length=0) = Field(..., description="A list of key skills from the job description that are not in the resume.")
    suggested_rewrites: conlist(RewrittenBulletPoint, min_length=0) = Field(..., description="A list of suggested bullet point rewrites.")
    tailored_summary: str = Field(..., description="A newly generated professional summary tailored to the job description.")