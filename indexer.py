from utils import build_embedding_index

if __name__ == "__main__":
    build_embedding_index()
    print("✅ Index built: artifacts/embeddings.npy + metadata.json")
