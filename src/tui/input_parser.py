"""Input parsing — extract @image references from user input."""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_IMG_PATH_RE = re.compile(r"@(\S+)")


# 返回值：
# [
#     {
#         "type": "image",
#         "source": {"type": "base64", "media_type": "image/png", "data": "base64编码数据..."}
#     },
#     {
#         "type": "text", 
#         "text": "其余的文本内容"
#     }
# ]
def parse_input(text: str) -> str | list:
    """Parse user input, extracting @path image references into content blocks.

    Returns plain string if no images, or a list of content blocks if images found.
    """
    matches = list(_IMG_PATH_RE.finditer(text)) # 正则表达式查找所有 @路径 格式的匹配项
    if not matches:
        return text

    image_blocks = []
    for m in matches:
        fpath = Path(m.group(1))
        if not fpath.suffix.lower() in _IMAGE_EXTS: # 检查扩展名
            continue
        if not fpath.exists():  # 检查文件是否存在
            continue
        # 获取媒体类型，并将文件内容编码为base64字符串
        media_type = mimetypes.guess_type(str(fpath))[0] or "image/png"
        data = base64.standard_b64encode(fpath.read_bytes()).decode("ascii")
        image_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })

    if not image_blocks:
        return text

    # 移除原文中的 @路径 标记，并将剩余文本与图片块组合成内容列表
    cleaned = _IMG_PATH_RE.sub("", text).strip()
    content: list[dict] = list(image_blocks)
    if cleaned:
        content.append({"type": "text", "text": cleaned})
    return content
