# Retrieval side of RAG: question -> MiniLM embedding -> FAISS search -> real
# Wikipedia passages -> prompt -> DarkGLM answer.
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

from model.config import DarkGLMConfig
from model.model import DarkGLM

RAG_DIR = Path("rag")
INDEX_PATH = RAG_DIR / "wikipedia.index"
PASSAGES_PATH = RAG_DIR / "wikipedia_passages.txt"
STATE_PATH = RAG_DIR / "build_state.json"
OFFSETS_PATH = RAG_DIR / "passage_offsets.npy"

TOP_K = 3
SEARCH_NPROBE = 16          # must match the build; raising it trades speed for accuracy

# Newest checkpoint that understands the User:/Assistant: format, best first.
CHECKPOINT_CANDIDATES = [
    "checkpoints/combined_best_model.pt",
    "checkpoints/strategyqa_best_model.pt",
    "checkpoints/dolly_best_model.pt",
]

# CPU on purpose: a training run usually has the GPU near its 8GB ceiling, and
# loading a second copy of the model onto it would risk OOM-ing that run.
DEVICE = "cpu"
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.3
TOP_P_K = 50
REPETITION_PENALTY = 1.3
ANSWER_TOKEN_RESERVE = MAX_NEW_TOKENS + 16

# Training examples are full of these labels, so left alone the model just keeps
# writing the document and invents its own next turn.
#STOP_SEQUENCES = ("\nUser:", "\nContext:", "\nAssistant:")

DEFAULT_QUESTIONS = [
    "Can cows fly?",
]


def load_index():
    if not (INDEX_PATH.exists() and PASSAGES_PATH.exists() and STATE_PATH.exists()):
        raise SystemExit(
            f"Index not built yet - run `python -m rag.build_index` first "
            f"(looked for {INDEX_PATH}, {PASSAGES_PATH}, {STATE_PATH})."
        )
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as error:
        raise SystemExit(
            f"Could not read {INDEX_PATH}: {error}\n"
            f"If rag/build_index.py is running it may have been caught mid-checkpoint - "
            f"wait a moment and try again."
        )
    index.nprobe = SEARCH_NPROBE
    return index


def build_line_offsets(path, line_count):
    # Vector N in the index corresponds to line N of the passages file, so a byte
    # offset per line is enough to look any passage up without holding the 1.4GB
    # file in memory. The file may still be growing - stop at the indexed count.
    offsets = np.zeros(line_count, dtype=np.int64)
    written = 0
    offset = 0
    with open(path, "rb") as f:
        for line in f:
            if written >= line_count:
                break
            offsets[written] = offset
            written += 1
            offset += len(line)
    if written < line_count:
        raise SystemExit(
            f"{path} has only {written:,} lines but the index holds {line_count:,} "
            f"vectors - the two are out of sync, rebuild the index."
        )
    return offsets


def load_line_offsets(path, line_count):
    if OFFSETS_PATH.exists():
        cached = np.load(OFFSETS_PATH, mmap_mode="r")
        if len(cached) >= line_count:
            return cached
        # Windows refuses to overwrite a file that is still memory-mapped, and the
        # cache goes stale every time the index grows.
        cached._mmap.close()
        del cached

    print("Mapping passage line offsets (one-off scan, cached afterwards)...")
    offsets = build_line_offsets(path, line_count)
    np.save(OFFSETS_PATH, offsets)
    return offsets


def read_passage(passages_file, offsets, position):
    passages_file.seek(int(offsets[position]))
    return passages_file.readline().decode("utf-8").rstrip("\n")


def retrieve(question, embedder, index, passages_file, offsets, top_k=TOP_K):
    query_vector = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)

    # search returns one row per query; [0] unwraps the single question's results.
    scores, positions = index.search(query_vector, top_k)
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:  # fewer than top_k matches found in the probed clusters
            continue
        results.append((float(score), read_passage(passages_file, offsets, position)))
    return results


def build_prompt(question, results, tokenizer, context_length):
    question_block = f"User: {question}\n\nContext: "
    budget = context_length - ANSWER_TOKEN_RESERVE - len(tokenizer.encode(question_block).ids)

    chosen = []
    for _, passage in results:
        cost = len(tokenizer.encode(passage + " ").ids)
        if cost > budget:
            continue
        chosen.append(passage)
        budget -= cost

    context = " ".join(chosen) if chosen else "(no relevant passage found)"
    return f"User: {question}\n\nContext: {context}\nAssistant:", len(chosen)


@torch.no_grad()
def generate(model, tokenizer, prompt, config):
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=DEVICE)
    prompt_length = input_ids.shape[1]
    endoftext_id = tokenizer.token_to_id("<|endoftext|>")

    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids[:, -config.context_length:])[:, -1, :] / TEMPERATURE

        # Penalise only what the model itself has said. Including the prompt would
        # punish it for reusing words from the retrieved passage, which is exactly
        # what a grounded answer has to do.
        for token_id in set(input_ids[0, prompt_length:].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= REPETITION_PENALTY
            else:
                logits[0, token_id] *= REPETITION_PENALTY

        top_values, top_indices = torch.topk(logits, TOP_P_K)
        probs = torch.softmax(top_values, dim=-1)
        next_token = top_indices.gather(-1, torch.multinomial(probs, num_samples=1))
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == endoftext_id:
            break

        answer = tokenizer.decode(input_ids[0, prompt_length:].tolist())
        """if any(stop in answer for stop in STOP_SEQUENCES):
            break"""

    answer = tokenizer.decode(input_ids[0, prompt_length:].tolist())
    """for stop in STOP_SEQUENCES:
        answer = answer.split(stop)[0]"""
    return answer.strip()


def main():
    questions = sys.argv[1:] or DEFAULT_QUESTIONS

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    index = load_index()
    print(
        f"Index: {index.ntotal:,} passages from {state['article_count']:,} articles "
        f"(build still in progress unless that matches the full corpus)"
    )

    offsets = load_line_offsets(PASSAGES_PATH, index.ntotal)
    passages_file = open(PASSAGES_PATH, "rb")

    print("Loading MiniLM embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    checkpoint = next((c for c in CHECKPOINT_CANDIDATES if Path(c).exists()), None)
    if checkpoint is None:
        raise SystemExit(f"No usable checkpoint found - looked for {CHECKPOINT_CANDIDATES}")

    config = DarkGLMConfig()
    model = DarkGLM(config).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    print(f"Loaded {checkpoint} on {DEVICE}")

    try:
        for question in questions:
            print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")

            results = retrieve(question, embedder, index, passages_file, offsets)
            for rank, (score, passage) in enumerate(results, start=1):
                preview = passage if len(passage) <= 300 else passage[:300] + "..."
                print(f"\n[{rank}] similarity {score:.3f}\n{preview}")

            prompt, used = build_prompt(question, results, tokenizer, config.context_length)
            print(f"\n-- prompt uses {used} of {len(results)} retrieved passages --")

            answer = generate(model, tokenizer, prompt, config)
            print(f"\nDarkGLM: {answer}")
    finally:
        passages_file.close()


if __name__ == "__main__":
    main()
