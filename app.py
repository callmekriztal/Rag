from src.search import RAGSearch


def main():
    rag = RAGSearch()

    query = "What is attention mechanism?"

    answer = rag.search_and_answer(
        query=query,
        top_k=3,
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()