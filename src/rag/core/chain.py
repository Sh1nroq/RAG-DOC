from openai import OpenAI


def get_answer_llm(openai_client: OpenAI, context: list[str], query: str) -> str:
    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question based only on the provided context. "
        "If the answer is not in the context, say you don't have enough information. "
        "Be concise and precise."
    )

    context_str = "\n".join(context)

    completion = openai_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {query}",
            },
        ],
    )
    return completion.choices[0].message.content or ""
