from app.teacher.teacher_llm import get_teacher_llm, MockTeacher, TeacherResponse

def test_mock_teacher_generation():
    teacher = MockTeacher(model_name="mock-gpt-70b")
    resp = teacher.generate_summary("T001", "Customer unable to login to application.")

    assert isinstance(resp, TeacherResponse)
    assert resp.ticket_id == "T001"
    assert resp.teacher_model == "mock-gpt-70b"
    assert resp.provider == "mock"
    assert "Customer unable to login to application" in resp.teacher_summary
    assert resp.latency_ms >= 0.0

def test_teacher_factory_mock():
    teacher = get_teacher_llm(provider_override="mock")
    assert isinstance(teacher, MockTeacher)
    assert teacher.provider_name == "mock"
