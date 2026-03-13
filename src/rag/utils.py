from rag.core.chain import get_answer_llm


def main() -> None:
    message = input()
    print(get_answer_llm(message))


if __name__ == "__main__":
    main()
