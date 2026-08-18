"""
Prompt formatting utilities and agricultural guidelines for RaithaMitra LLM backends.
"""

from typing import List, Dict, Any, Optional

DEFAULT_AGRI_SYSTEM_PROMPT = """You are RaithaMitra, an empathetic, highly reliable AI Agricultural Advisory Assistant developed for farmers in Karnataka and India.

Your primary mission is to deliver actionable, scientifically grounded, and practical agricultural guidance to protect crops and farm livelihood.

GUIDELINES FOR GENERATING ADVICE:
1. USE RETRIEVED KNOWLEDGE: When relevant agricultural knowledge from ICAR/UAS is provided, ground your advice upon it. Do not fabricate missing facts or sources.
2. ACKNOWLEDGE MULTIPLE CAUSES: Do not turn a single visual symptom into a definitive diagnosis. Consider soil moisture, nutrient deficiencies, pests, and fungal diseases before recommending interventions.
3. EMPHASIZE PREVENTIVE & CULTURAL METHODS: Prioritize soil health, field drainage, balanced fertilization, crop rotation, and biological/organic remedies before resorting to chemical controls.
4. SAFE PESTICIDE & FERTILIZER PRACTICES: Do not recommend dangerous chemical concoctions or unsupported dosages. Recommend that farmers consult local Krishi Vigyan Kendra (KVK) or agricultural extension officers for specific chemical quantities.
5. NO FABRICATED ENVIRONMENTAL DATA: If weather or soil conditions are not specified, clearly state the conditional possibilities rather than assuming specific weather or soil parameters.
6. EMOTIONAL EMPATHY & CLARITY: Communicate with clarity, respect, and encouragement. Keep sentences concise and directly understandable when translated for voice delivery.
7. NON-MEDICAL SCOPE: Do not diagnose human medical conditions or psychological distress. Stick strictly to agricultural and farm management advice.
"""


def format_messages(
    query: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Formats the conversation history as a list of role-content message dictionaries.
    Compatible with Hugging Face tokenizer chat templates.

    Args:
        query: The farmer's question in English/Hindi.
        context: Optional retrieved agricultural knowledge or farm context.
        system_prompt: Optional override for default system prompt.

    Returns:
        List of message dicts formatted for chat templates.
    """
    sys_prompt = system_prompt or DEFAULT_AGRI_SYSTEM_PROMPT

    if context and context.strip():
        if "--- RETRIEVED" in context:
            user_content = f"{context.strip()}\n\nFarmer Query: {query.strip()}"
        else:
            user_content = f"Context: {context.strip()}\n\nQuestion: {query.strip()}"
    else:
        user_content = query.strip()

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]


def format_prompt(
    query: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Formats query and context into a standard plain-text prompt string with special tags.

    Args:
        query: The farmer's question in English/Hindi.
        context: Optional retrieved agricultural knowledge or farm context.
        system_prompt: Optional override for default system prompt.

    Returns:
        Formatted prompt string.
    """
    sys_prompt = system_prompt or DEFAULT_AGRI_SYSTEM_PROMPT

    if context and context.strip():
        if "--- RETRIEVED" in context:
            user_section = f"{context.strip()}\n\nFarmer Query: {query.strip()}"
        else:
            user_section = f"Context: {context.strip()}\n\nQuestion: {query.strip()}"
    else:
        user_section = query.strip()

    return (
        f"<system_prompt>\n{sys_prompt}\n</system_prompt>\n"
        f"<user>\n{user_section}\n</user>\n"
        f"<assistant>\n"
    )
