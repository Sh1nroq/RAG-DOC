from src.rag.llm_api import get_answer_llm


def main():
    message = input()
    print(get_answer_llm(message))

if __name__ == "__main__":
    main()