from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")


documents = [
    {
        "title": "Python",
        "content": "Python is a high-level programming language used for software development, automation, data science, machine learning and artificial intelligence."
    },
    {
        "title": "Artificial Intelligence",
        "content": "Artificial intelligence enables machines to perform tasks that normally require human intelligence. Machine learning and neural networks are important areas of AI."
    },
    {
        "title": "Computer Networking",
        "content": "Computer networking allows computers and devices to communicate with each other using protocols such as TCP, IP, HTTP and DNS."
    },
    {
        "title": "Database",
        "content": "A database is used to store, organize and manage information. Relational databases use tables, rows and columns."
    },
    {
        "title": "Web Development",
        "content": "Web development is the process of creating websites and web applications using technologies such as HTML, CSS and JavaScript."
    }
]


texts = [
    document["content"]
    for document in documents
]


print("Creating document embeddings...")
document_embeddings = model.encode(texts)

print("Document embeddings created.")


def semantic_search(query, top_k=5):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    results = []

    for index, similarity in enumerate(similarities):

        results.append({
            "title": documents[index]["title"],
            "content": documents[index]["content"],
            "semantic_score": round(
                float(similarity),
                4
            )
        })

    results.sort(
        key=lambda item: item["semantic_score"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    query = input(
        "\nEnter your search query: "
    )

    results = semantic_search(query)

    print("\nSemantic Search Results")
    print("=" * 50)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{rank}. {result['title']}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score']}"
        )

        print(
            result["content"]
        )