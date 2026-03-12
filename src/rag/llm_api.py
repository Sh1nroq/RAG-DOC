from dotenv import load_dotenv
from openai import OpenAI


def get_answer_llm(message: str) -> str:
    load_dotenv()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", max_retries=5)

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[{"role": "user", "content": message}],
    )
    return completion.choices[0].message.content or ""
