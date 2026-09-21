"""Teacher LLM integration module for gold-standard supervision labels and distillation."""
from app.teacher.teacher_llm import TeacherLLM, get_teacher_llm, TeacherResponse

__all__ = ["TeacherLLM", "get_teacher_llm", "TeacherResponse"]
