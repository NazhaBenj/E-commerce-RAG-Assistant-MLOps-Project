from src.rag.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(top_k=3)
results = retriever.retrieve("Comment annuler une commande sur Jumia Maroc ?")

for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['chunk_id']}")
    print("Sujet :", result["metadata"].get("topic"))
    print("Score fusion :", round(result["fusion_score"], 5))
    print("Texte :", result["text"][:350])
