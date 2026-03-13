from openai import OpenAI

from rag.config import settings


def get_answer_llm(message: str) -> str:

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENAI_API_KEY,
        max_retries=5,
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[{"role": "user", "content": message}],
    )
    return completion.choices[0].message.content or ""
