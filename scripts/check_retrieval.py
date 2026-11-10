from src.rag.retriever import Retriever

retriever = Retriever(top_k=3)

question = "Comment annuler une commande sur Jumia Maroc ?"
results = retriever.retrieve(question)

print("\n=== RÉSULTATS DE RECHERCHE ===")

for i, result in enumerate(results, start=1):
    print(f"\n[{i}] {result['chunk_id']}")
    print(f"Distance : {result['distance']:.4f}")
    print(f"Document : {result['metadata'].get('document_id')}")
    print(f"Sujet : {result['metadata'].get('topic')}")
    print(f"Source : {result['metadata'].get('source_url')}")
    print(f"Texte : {result['text'][:500]}")
