"""技能系统 —— 加载、注册并执行基于 SKILL.md 的技能。

参照 claude-code 的 src/skills/loadSkillsDir.ts 和
src/tools/SkillTool/SkillTool.ts 设计。

技能是带有 YAML 前置元数据（frontmatter）的 Markdown 文件，用于定义可复用的提示词（prompts）。
它们可以分为：
  内置（Bundled） —— 通过 register_skill() 在代码中注册
  项目级（Project） —— 从 .cc-mini/skills/<name>/SKILL.md 发现
  用户级（User） —— 从 ~/.cc-mini/skills/<name>/SKILL.md 发现

执行模式：
  内联（inline）（默认）：将提示词注入到当前对话中
  分叉（fork）：提示词在隔离的回合中运行（消息会被保存/恢复）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Skill definition
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """A single skill definition."""
    name: str                                    # 技能名称
    description: str = ""                        # 技能描述
    when_to_use: str = ""                        # 使用场景说明，描述何时应该使用此技能
    user_invocable: bool = True                  # 是否允许用户直接调用此技能
    disable_model_invocation: bool = False       # 是否禁用模型自动调用此技能
    allowed_tools: list[str] = field(default_factory=list)  # 此技能允许使用的工具列表
    model: str | None = None                     # 指定用于执行此技能的模型
    context: str = "inline"                      # 执行上下文类型，"inline"(内联)或"fork"(分支)
    argument_hint: str = ""                      # 调用此技能时参数的提示信息
    paths: list[str] = field(default_factory=list)  # Git忽略风格的路径模式，用于指定技能相关的文件路径
    source: str = "project"                      # 技能来源，"bundled"(捆绑的内置技能), "project"(项目技能), "user"(用户自定义技能)
    skill_root: str | None = None                # 技能根目录，用于$SKILL_DIR变量替换的基础目录

    # The prompt content (body of SKILL.md, after frontmatter)
    _prompt_text: str = ""                       # 提示内容（SKILL.md文件主体内容，去除前置元数据后）
    # Or a dynamic prompt generator (for bundled skills)
    _prompt_fn: Callable[[str], str] | None = None  # 动态提示生成器（用于内置技能）

    def get_prompt(self, args: str = "") -> str:
        """Return the final prompt text, substituting variables."""
        if self._prompt_fn is not None:                 # 如果有动态提示生成器，则使用它来生成提示文本
            return self._prompt_fn(args)
        text = self._prompt_text
        # Variable substitution (matches claude-code's processPromptSlashCommand)
        text = text.replace("$ARGUMENTS", args)
        if self.skill_root:
            text = text.replace("${CLAUDE_SKILL_DIR}", self.skill_root)
        if args and self.argument_hint:
            text = text.replace(f"${{{self.argument_hint}}}", args)     # 如果提供了参数且有参数提示，则替换特定的参数提示占位符

        return text


# ---------------------------------------------------------------------------
# YAML frontmatter parser (minimal, no PyYAML dependency)
# ---------------------------------------------------------------------------

# 文档前言部分（frontmatter）
# 前言部分通常位于文档开头，以 "---" 开始和结束，中间包含YAML格式的元数据
# 例如：
# ---
# title: 示例标题
# author: 作者名
# ---
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)   # 捕获两个 --- 之间所有的内容


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``text`` into (frontmatter_dict, body).

    Uses a minimal key: value parser — supports strings, booleans, and
    comma-separated lists.  Does not handle nested YAML.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    raw = m.group(1)            # 捕获内容
    body = text[m.end():]       # 匹配到的完整文本
    meta: dict[str, Any] = {}   # 剩余部分

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower().replace("-", "_")
        val = val.strip()
        # Boolean
        if val.lower() in ("true", "yes"):
            meta[key] = True
        elif val.lower() in ("false", "no"):
            meta[key] = False
        # List (comma-separated)
        elif "," in val:
            meta[key] = [v.strip() for v in val.split(",") if v.strip()]
        # Quoted string
        elif (val.startswith('"') and val.endswith('"')) or \
             (val.startswith("'") and val.endswith("'")):
            meta[key] = val[1:-1]
        else:
            meta[key] = val

    return meta, body


def _ensure_str(val: Any, default: str = "") -> str:    # 将给定的值转换为字符串，如果值是列表，则将列表元素用逗号连接后返回字符串
    """Coerce *val* to a string — rejoin lists produced by the frontmatter parser."""
    if val is None:
        return default
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def _skill_from_frontmatter(meta: dict[str, Any], body: str,
                             name: str, source: str,
                             skill_root: str | None = None) -> Skill:
    """
    Build a ``Skill`` from parsed frontmatter and body text.
    在_parse_frontmatter之后处理
    """

    # 强制执行一次以逗号为分隔符的列表构建与空白符清理
    allowed = meta.get("allowed_tools", [])
    if isinstance(allowed, str):
        allowed = [t.strip() for t in allowed.split(",") if t.strip()]

    paths = meta.get("paths", [])
    if isinstance(paths, str):
        paths = [p.strip() for p in paths.split(",") if p.strip()]

    return Skill(
        name=_ensure_str(meta.get("name"), name),
        description=_ensure_str(meta.get("description")),
        when_to_use=_ensure_str(meta.get("when_to_use")),
        user_invocable=meta.get("user_invocable", True),
        disable_model_invocation=meta.get("disable_model_invocation", False),
        allowed_tools=allowed,
        model=meta.get("model"),
        context=_ensure_str(meta.get("context"), "inline"),
        argument_hint=_ensure_str(meta.get("arguments")),
        paths=paths,
        source=source,
        skill_root=skill_root,
        _prompt_text=body.strip(),
    )


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Skill] = {}


def register_skill(skill: Skill) -> None:
    """Add a skill to the global registry."""
    _REGISTRY[skill.name] = skill


def get_skill(name: str) -> Skill | None:
    """Look up a skill by name."""
    return _REGISTRY.get(name)


def list_skills(user_invocable_only: bool = True) -> list[Skill]:
    """Return all registered skills, optionally filtered."""
    skills = list(_REGISTRY.values())
    if user_invocable_only:
        skills = [s for s in skills if s.user_invocable]
    return sorted(skills, key=lambda s: (s.source != "bundled", s.name))


def clear_skills(source: str | None = None) -> None:
    """Remove skills from the registry.  If *source* given, only that source."""
    if source is None:
        _REGISTRY.clear()
    else:
        to_remove = [k for k, v in _REGISTRY.items() if v.source == source]
        for k in to_remove:
            del _REGISTRY[k]


# ---------------------------------------------------------------------------
# Skill discovery from disk
# ---------------------------------------------------------------------------

def load_skills_from_dir(skills_dir: Path, source: str = "project") -> list[Skill]:
    """
        扫描 skills_dir 以查找 <name>/SKILL.md 并注册每个技能。

        匹配 claude-code 的 loadSkillsDir.ts 目录格式加载逻辑：
        仅识别包含 SKILL.md 文件的目录。

        同时支持直接位于该目录下的单个 .md 文件（兼容 claude-code 遗留的
        /commands/ 格式）。
    """
    loaded: list[Skill] = []
    if not skills_dir.is_dir():
        return loaded

    for entry in sorted(skills_dir.iterdir()):  # 对目录下的子项进行遍历
        skill = None
        if entry.is_dir():
            skill_md = entry / "SKILL.md"       # 探寻子目录下的 SKILL.md 文件
            if not skill_md.exists():
                # Fallback: look for any .md file in the directory
                md_files = list(entry.glob("*.md"))
                if md_files:
                    skill_md = md_files[0]
                else:
                    continue
            try:
                text = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            meta, body = _parse_frontmatter(text)
            skill = _skill_from_frontmatter(
                meta, body,
                name=entry.name,
                source=source,
                skill_root=str(entry),
            )
        elif entry.suffix == ".md" and entry.is_file(): # 当遍历到的 entry 是一个文件且后缀为 .md
            # Legacy single-file format
            try:
                text = entry.read_text(encoding="utf-8")
            except Exception:
                continue
            meta, body = _parse_frontmatter(text)
            skill = _skill_from_frontmatter(
                meta, body,
                name=entry.stem,
                source=source,
                skill_root=str(entry.parent),
            )

        if skill and skill._prompt_text:
            register_skill(skill)
            loaded.append(skill)

    return loaded


def discover_skills(cwd: str | None = None) -> list[Skill]:
    """Discover and register skills from standard locations.

    Search order (matches claude-code's four-tier hierarchy):
      1. Bundled skills (already registered via ``register_bundled_skills()``)
      2. User skills:    ``~/.cc-mini/skills/``
      3. Project skills: ``{cwd}/.cc-mini/skills/``

    Returns newly loaded skills (excludes already-registered bundled ones).
    """
    loaded: list[Skill] = []

    # 用户级技能
    user_dir = Path.home() / ".cc-mini" / "skills"
    loaded.extend(load_skills_from_dir(user_dir, source="user"))

    # 项目级技能
    if cwd:
        project_dir = Path(cwd) / ".cc-mini" / "skills"
        loaded.extend(load_skills_from_dir(project_dir, source="project"))

    return loaded


# ---------------------------------------------------------------------------
# System prompt section
# ---------------------------------------------------------------------------

def build_skills_prompt_section() -> str:
    """Build the skills listing for the system prompt.

    Matches claude-code's ``SkillTool/prompt.ts`` — lists available skills
    so the model knows what it can invoke via ``/skill-name``.
    """
    skills = list_skills(user_invocable_only=False) # 从全局注册表中提取所有当前可用的技能实例，参数设定为 False，意味着它将提取所有注册的技能，无论其是否配置为仅供用户手动触发。
    if not skills:
        return ""

    lines = ["# Available Skills", ""]
    for s in skills:
        desc = s.description or "(no description)"
        line = f"- /{s.name}: {desc}"
        if s.when_to_use:
            line += f" — {s.when_to_use}"
        lines.append(line)

    return "\n".join(lines)
