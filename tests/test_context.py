from unittest.mock import patch, MagicMock
import subprocess
from core.context import build_system_prompt, build_system_prompt_layout, _get_git_section, _get_claude_md_section
from features.memory import build_working_memory_section, build_working_memory_snapshot


def test_build_system_prompt_contains_base_instructions():
    prompt = build_system_prompt(cwd="/tmp")
    assert "software engineering tasks" in prompt
    assert "tools" in prompt.lower()


def test_build_system_prompt_contains_env_info():
    prompt = build_system_prompt(cwd="/tmp")
    assert "Primary working directory: /tmp" in prompt
    assert "Platform:" in prompt
    assert "Shell:" in prompt


def test_build_system_prompt_contains_working_directory():
    prompt = build_system_prompt(cwd="/some/test/dir")
    assert "/some/test/dir" in prompt


def test_build_system_prompt_includes_git_status_when_available():
    fake_result = MagicMock()
    fake_result.stdout = "main"

    with patch("core.context.subprocess.run", return_value=fake_result):
        prompt = build_system_prompt(cwd="/tmp")
    assert "Git Status" in prompt
    assert "main" in prompt


def test_build_system_prompt_includes_claude_md(tmp_path):
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Test Project\nSome instructions here.")

    prompt = build_system_prompt(cwd=str(tmp_path))
    assert "CLAUDE.md" in prompt
    assert "Test Project" in prompt


def test_build_system_prompt_without_claude_md(tmp_path):
    prompt = build_system_prompt(cwd=str(tmp_path))
    # Should not have the CLAUDE.md section header (beyond the base prompt)
    assert "# Test Project" not in prompt


def test_get_git_section_returns_branch_and_log(tmp_path):
    def fake_run(cmd, **kwargs):
        result = MagicMock()
        if "branch" in cmd:
            result.stdout = "feature-branch"
        elif "status" in cmd:
            result.stdout = " M file.py"
        elif "log" in cmd:
            result.stdout = "abc1234 some commit"
        else:
            result.stdout = ""
        return result

    with patch("core.context.subprocess.run", side_effect=fake_run):
        status = _get_git_section(str(tmp_path))

    assert "feature-branch" in status
    assert "M file.py" in status
    assert "abc1234" in status


def test_get_git_section_returns_empty_on_non_git_dir():
    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = ""
        return result

    with patch("core.context.subprocess.run", side_effect=fake_run):
        status = _get_git_section("/tmp/not-a-git-repo")
    assert status == ""


def test_get_git_section_returns_empty_on_exception():
    with patch("core.context.subprocess.run", side_effect=OSError("fail")):
        status = _get_git_section("/tmp")
    assert status == ""


def test_get_claude_md_section_reads_file(tmp_path):
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("hello world")

    result = _get_claude_md_section(str(tmp_path))
    assert "hello world" in result
    assert "CLAUDE.md" in result


def test_get_claude_md_section_returns_empty_when_missing(tmp_path):
    result = _get_claude_md_section(str(tmp_path))
    assert result == ""


def test_get_claude_md_section_truncates_large_file(tmp_path):
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("x" * 20_000)

    result = _get_claude_md_section(str(tmp_path))
    # Section includes header, so content is truncated to fit within 10k chars
    assert len(result) <= 10_100  # Allow some margin for the header


def test_build_system_prompt_layout_exposes_sections(tmp_path):
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Test Project\nSome instructions here.")

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        if "branch" in cmd:
            result.stdout = "feature-branch"
        elif "status" in cmd:
            result.stdout = " M file.py"
        elif "log" in cmd:
            result.stdout = "abc1234 some commit"
        else:
            result.stdout = ""
        result.returncode = 0
        return result

    with patch("core.context.subprocess.run", side_effect=fake_run):
        layout = build_system_prompt_layout(cwd=str(tmp_path), model="test-model")

    assert layout["prompt"].startswith(layout["stable_prefix"])
    assert "# Environment" not in layout["stable_prefix"]
    assert layout["stats"]["section_count"] == len(layout["sections"])
    assert layout["stats"]["stable_section_count"] == 7
    assert layout["stats"]["dynamic_section_count"] == 3
    assert layout["stats"]["total_chars"] == len(layout["prompt"])
    assert layout["stats"]["stable_chars"] == len(layout["stable_prefix"])

    names = [section["name"] for section in layout["sections"]]
    assert names == [
        "intro",
        "system",
        "doing_tasks",
        "actions",
        "using_tools",
        "tone_and_style",
        "output_efficiency",
        "environment",
        "git",
        "claude_md",
    ]
    assert all(section["stable"] for section in layout["sections"][:7])
    assert not any(section["stable"] for section in layout["sections"][7:])


def test_build_system_prompt_layout_keeps_stable_prefix_constant():
    fake_result = MagicMock()
    fake_result.stdout = ""
    fake_result.returncode = 0

    with patch("core.context.subprocess.run", return_value=fake_result):
        layout_a = build_system_prompt_layout(cwd="/tmp/project-a", model="model-a")
        layout_b = build_system_prompt_layout(cwd="/tmp/project-b", model="model-b")

    assert layout_a["stable_prefix"] == layout_b["stable_prefix"]
    assert layout_a["prompt"] != layout_b["prompt"]


def test_build_working_memory_snapshot_tracks_recent_messages():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
    ]

    snapshot = build_working_memory_snapshot(messages)

    assert snapshot["last_user"] == "third question"
    assert snapshot["last_assistant"] == "second answer"
    assert len(snapshot["recent_messages"]) == 4
    assert snapshot["recent_messages"][-1] == {"role": "user", "text": "third question"}
    assert "User: third question" in snapshot["summary"]


def test_build_working_memory_snapshot_records_worker_notification():
    snapshot = build_working_memory_snapshot(
        [{"role": "assistant", "content": "done"}],
        turn_artifact={
            "context": {"source": "worker_notification", "summary": "worker finished indexing"}
        },
    )

    assert "worker finished indexing" in snapshot["carry_forwards"]


def test_build_working_memory_section_renders_content():
    section = build_working_memory_section({
        "summary": "User: ask about cache | Assistant: explained cache",
        "carry_forwards": ["Auto-compact completed (100→40 tokens)"],
        "recent_messages": [
            {"role": "user", "text": "ask about cache"},
            {"role": "assistant", "text": "explained cache"},
        ],
    })

    assert "# Working Memory" in section
    assert "Carry-forwards:" in section
    assert "Recent conversation:" in section
    assert "User: ask about cache" in section


def test_build_system_prompt_layout_includes_working_memory_section():
    fake_result = MagicMock()
    fake_result.stdout = ""
    fake_result.returncode = 0

    with patch("core.context.subprocess.run", return_value=fake_result):
        layout = build_system_prompt_layout(
            cwd="/tmp/project",
            model="test-model",
            working_memory={
                "summary": "User: ask cache | Assistant: answered",
                "carry_forwards": ["worker finished indexing"],
                "recent_messages": [],
            },
        )

    names = [section["name"] for section in layout["sections"]]
    assert names[-1] == "working_memory"
    assert "# Working Memory" in layout["prompt"]
