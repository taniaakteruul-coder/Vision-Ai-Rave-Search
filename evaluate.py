import os
from PIL import Image
from utils import build_embedding_index, search_by_image

TEST_DIR = "data/test_queries"  # structure: data/test_queries/<category>/*.jpg

def main():
    # Build index first
    build_embedding_index()

    total = 0
    top1 = 0
    top5 = 0

    if not os.path.exists(TEST_DIR):
        print("No test folder found at data/test_queries/. Create it to run evaluation.")
        return

    for category in os.listdir(TEST_DIR):
        cat_dir = os.path.join(TEST_DIR, category)
        if not os.path.isdir(cat_dir):
            continue

        for fn in os.listdir(cat_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            total += 1
            img = Image.open(os.path.join(cat_dir, fn)).convert("RGB")
            results = search_by_image(img, top_k=5)

            # We treat correct if item_name contains the folder category term
            predicted_names = [(r.get("item_name") or "") for (r, _) in results]

            if predicted_names:
                if category.lower() in predicted_names[0].lower():
                    top1 += 1
                if any(category.lower() in n.lower() for n in predicted_names):
                    top5 += 1

    if total == 0:
        print("No test images found.")
        return

    print(f"Queries: {total}")
    print(f"Top-1 accuracy: {top1/total:.3f}")
    print(f"Top-5 accuracy: {top5/total:.3f}")

if __name__ == "__main__":
    main()
