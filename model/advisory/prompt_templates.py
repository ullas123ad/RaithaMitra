"""
RaithaMitra Agricultural Advisory Prompt Templates.

Defines standard persona instructions, prompt formatters, and context wrappers
for agricultural LLM backends (Dhenu, AgriParam, etc.).
"""

from typing import Optional, List, Dict, Any


DEFAULT_AGRI_SYSTEM_PROMPT = (
    "You are RaithaMitra's AI Agricultural Advisor, providing practical, scientifically grounded, "
    "and actionable advice to Indian farmers. Adhere strictly to the following guidelines:\n"
    "1. Provide practical, step-by-step agricultural recommendations in simple, understandable language.\n"
    "2. Avoid unnecessary technical jargon.\n"
    "3. Do not invent or assume live weather or soil conditions; ask for missing local details when relevant.\n"
    "4. Do not claim to have physically inspected the crop.\n"
    "5. Clearly indicate uncertainty if information is insufficient; do not present a tentative disease diagnosis as absolute certainty.\n"
    "6. Prioritize cultural, biological, and balanced nutrient practices before chemical interventions. If recommending chemical sprays, always include standard safety precautions and dilution rates.\n"
    "7. Keep responses concise, clear, and easy to understand for voice-based and mobile farmer interfaces."
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
    Formats the query into a single plaintext prompt string.

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
