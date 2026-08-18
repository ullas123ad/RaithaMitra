"""
RaithaMitra Agricultural Advisory Prompt Templates.

Defines standard persona instructions, prompt formatters, and context wrappers
for agricultural LLM backends (AgriParam, etc.).
"""

from typing import Optional, List, Dict, Any


DEFAULT_AGRI_SYSTEM_PROMPT = (
    "You are RaithaMitra's AI Agricultural Advisor. "
    "You provide practical, accurate, and scientifically grounded agricultural advice "
    "to Indian farmers regarding crop health, pest and disease management, soil health, "
    "irrigation practices, weather adaptation, and government farming schemes. "
    "Provide clear, concise, step-by-step actionable recommendations."
)


def format_messages(
    query: str,
    system_prompt: Optional[str] = None,
    context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Formats the user query into a standard list of chat messages.

    Args:
        query: The farmer's question or advisory request.
        system_prompt: Custom system instruction override.
        context: Optional additional domain context (e.g., crop type, region).

    Returns:
        List of message dictionaries with 'role' and 'content'.
    """
    sys_instruction = system_prompt.strip() if system_prompt else DEFAULT_AGRI_SYSTEM_PROMPT

    user_content = query.strip()
    if context and context.strip():
        user_content = f"Context: {context.strip()}\n\nQuestion: {user_content}"

    return [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": user_content}
    ]


def format_prompt(
    query: str,
    system_prompt: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Formats the query into a single plaintext prompt string suitable
    for completion or raw generation models.

    Args:
        query: The farmer's question.
        system_prompt: Custom system instruction override.
        context: Optional domain context.

    Returns:
        Formatted prompt string.
    """
    sys_instruction = system_prompt.strip() if system_prompt else DEFAULT_AGRI_SYSTEM_PROMPT

    user_content = query.strip()
    if context and context.strip():
        user_content = f"Context: {context.strip()}\n\nQuestion: {user_content}"

    return f"<system_prompt>\n{sys_instruction}\n</system_prompt>\n\n<user>\n{user_content}\n</user>\n\n<assistant>\n"
