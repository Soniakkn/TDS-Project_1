import os
import glob
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load and chunk course markdown ===
def load_markdown_chunks(folder="course_md"):
    chunks = []
    file_paths = glob.glob(os.path.join(folder, "*.md"))
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            lines = text.split('\n')
            buffer = []
            for line in lines:
                buffer.append(line)
                if len(buffer) >= 10:
                    chunks.append("\n".join(buffer))
                    buffer = []
            if buffer:
                chunks.append("\n".join(buffer))
    return chunks

# === Load and chunk discourse posts from multiple JSON files ===
def load_discourse_chunks(folder="discourse_json"):
    chunks = []
    file_paths = glob.glob(os.path.join(folder, "*.json"))
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)

                # ✅ Skip if not a dictionary
                if not isinstance(data, dict):
                    print(f"⚠️ Skipping {path}: Not a JSON object")
                    continue

                # ✅ Safely extract posts
                posts = data.get("post_stream", {}).get("posts", [])
                for post in posts:
                    content = post.get("cooked", "") or post.get("raw", "")
                    if content:
                        text = content.replace("<p>", "").replace("</p>", "").replace("<br>", "\n")
                        lines = text.strip().split("\n")
                        buffer = []
                        for line in lines:
                            buffer.append(line.strip())
                            if len(buffer) >= 10:
                                chunks.append("\n".join(buffer))
                                buffer = []
                        if buffer:
                            chunks.append("\n".join(buffer))
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
    return chunks

# === Combine markdown and discourse chunks ===
markdown_chunks = load_markdown_chunks("course_md")
discourse_chunks = load_discourse_chunks("discourse_json")
all_chunks = markdown_chunks + discourse_chunks

print(f"Total Chunks - Markdown: {len(markdown_chunks)}, Discourse: {len(discourse_chunks)}")

# === Get embeddings ===
embeddings = model.encode(all_chunks)

# === Save to FAISS index ===
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(embeddings)
faiss.write_index(index, "tds_index.faiss")

# === Save chunk content for retrieval
with open("tds_chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ Embedding complete and saved.")
