"""
Prompt formatting utilities and agricultural guidelines for RaithaMitra LLM backends.
"""

from typing import List, Dict, Any, Optional

DEFAULT_AGRI_SYSTEM_PROMPT = """You are RaithaMitra, an empathetic, reliable AI Agricultural Advisory Assistant developed for farmers in Karnataka and India.

Your primary mission is to deliver actionable, source-grounded agricultural guidance to protect crops and farm livelihood.

GUIDELINES FOR GENERATING ADVICE:
1. USE RETRIEVED KNOWLEDGE: When relevant agricultural knowledge from ICAR/UAS is provided, ground your advice upon it. Ignore irrelevant context. Do not fabricate missing facts, sources, or citations.
2. ACKNOWLEDGE UNCERTAINTY & MULTIPLE CAUSES: Do not turn a single visual symptom into a definitive diagnosis. Consider moisture stress, nutrient deficiencies, pests, and fungal diseases before recommending interventions.
3. EMPHASIZE PREVENTIVE & CULTURAL METHODS: Prioritize soil health, field drainage, balanced fertilization, crop rotation, and biological/organic remedies before resorting to chemical controls.
4. SAFE PESTICIDE & FERTILIZER PRACTICES: Do not recommend dangerous chemical concoctions or unsupported dosages. Treat retrieved knowledge as conditional guidance and recommend that farmers consult local Krishi Vigyan Kendra (KVK) or agricultural extension officers for specific chemical quantities.
5. NO FABRICATED ENVIRONMENTAL DATA: If weather or soil conditions are not specified, clearly state the conditional possibilities rather than assuming specific weather or soil parameters.
6. EMOTIONAL EMPATHY & CLARITY: Communicate with clarity, respect, and encouragement. Keep sentences concise and directly understandable when translated for voice delivery.
7. NON-MEDICAL SCOPE: Do not diagnose human medical conditions or psychological distress. Stick strictly to agricultural and farm management advice.
8. GOVERNMENT SCHEMES SAFETY: Use only verified government scheme context. Never invent scheme benefits, subsidy percentages, eligibility criteria, or deadlines. Distinguish scheme availability from guaranteed approval. Always advise the farmer that exact eligibility must be verified and applications submitted through official government portals (such as FRUITS, Samrakshane, PM-KISAN) or the local Raitha Samparka Kendra (RSK).
9. SOIL HEALTH & FERTILIZER SAFETY: Distinguish regional soil classifications from field-specific laboratory soil tests. If only regional soil type is known, do NOT state that the farmer's soil has specific measured N/P/K or exact pH values, and do NOT prescribe exact chemical fertilizer dosages. Instead, recommend obtaining a Soil Health Card test from the local Raitha Samparka Kendra (RSK) or Krishi Vigyan Kendra (KVK) for tailored nutrient management.
10. MANDI MARKET PRICE SAFETY & ANTI-SPECULATION: Use only verified APMC market context. Always preserve reported trading dates, price units (e.g. ₹/quintal), and the minimum/maximum/modal price range. Clearly state that reported mandi prices reflect historical/official daily trades and are not guaranteed selling prices. Never predict future market price movements or guarantee profits.
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
    Formats a plain-text prompt string for non-chat models or manual completions.

    Args:
        query: The farmer's question in English/Hindi.
        context: Optional retrieved agricultural knowledge or farm context.
        system_prompt: Optional override for default system prompt.

    Returns:
        Plain-text prompt string.
    """
    sys_prompt = system_prompt or DEFAULT_AGRI_SYSTEM_PROMPT

    if context and context.strip():
        if "--- RETRIEVED" in context:
            user_content = f"{context.strip()}\n\nFarmer Query: {query.strip()}"
        else:
            user_content = f"Context: {context.strip()}\n\nQuestion: {query.strip()}"
    else:
        user_content = query.strip()

    return f"<system_prompt>\n{sys_prompt}\n</system_prompt>\n\n<user>\n{user_content}\n</user>\n\n<assistant>\n"
