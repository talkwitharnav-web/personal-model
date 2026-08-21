from pathlib import Path
import os

# Serve from the local cache and never contact the network. Everything the pipeline needs is
# already on disk - the MiniLM embedder is in the HuggingFace cache (174.7 MB), the FAISS index,
# passages, router and tokenizer are all in the repo. Without this, sentence-transformers still
# reaches out to check the model for updates on every start: harmless when online, a slow hang or
# an outright failure when not.
#
# Verified 2026-08-21: with these set the whole pipeline loads in 14.5 s and answers correctly
# with retrieval working, on a machine that cannot reach the network.
#
# ⚠ Only set if not already present, so `HF_HUB_OFFLINE=0` still works for anyone who needs to
# fetch a new dataset - the data/prepare_*.py scripts DO download, and they must stay able to.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RAG_DIR = Path("rag")
INDEX_PATH = RAG_DIR / "wikipedia.index"
PASSAGES_PATH = RAG_DIR / "wikipedia_passages.txt"
STATE_PATH = RAG_DIR / "build_state.json"
OFFSETS_PATH = RAG_DIR / "passage_offsets.npy"

ROUTER_PATH = Path("router/router.joblib")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
# v2 was refit on the real corpus (+9.42% compression). A checkpoint and its tokenizer are a
# MATCHED PAIR - serving a v2 model with the v1 tokenizer produces fluent nonsense, because the
# same ids mean different pieces of text. Generator picks between these by inspecting the
# checkpoint, so nothing has to be configured by hand.
TOKENIZER_PATH_V2 = Path("tokenizer/tokenizer_v2.json")

EMBEDDER_NAME = "all-MiniLM-L6-v2"

# Newest checkpoint that understands the User:/Assistant: format, best first.
# squad_v2_best_model.pt is deliberately excluded - it collapsed into refusing
# every question, because its checkpoint was chosen on a validation split that is
# 50% unanswerable while training was 33%.
CHECKPOINT_CANDIDATES = [
    "checkpoints/finetune_v2_best.pt",
    "checkpoints/combined_best_model.pt",
    "checkpoints/strategyqa_best_model.pt",
    "checkpoints/dolly_best_model.pt",
]

TOP_K = 3
CANDIDATE_K = 12          # extra candidates so boilerplate can be dropped
SEARCH_NPROBE = 16        # clusters scanned per query; higher is slower but more accurate

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.3
# Measured with tools/temperature_sweep.py: on factual questions the answer is
# identical from 0.1 to 0.3 and degrades badly by 0.6-0.9 ("a resonant transformer
# circuit" became "amerotherapy"). So these stay low; only chitchat gets any slack.
TEMPERATURE_BY_CATEGORY = {
    "maths": 0.1,
    "factual": 0.2,
    "instruction": 0.3,
    "reasoning": 0.3,
    "commonsense": 0.3,
    "summarise": 0.3,
    "chitchat": 0.5,
}
TOP_P_K = 50
REPETITION_PENALTY = 1.3
# Applied to the tokens of the model's own recent replies, so a conversation cannot fall into
# repeating one sentence forever. Deliberately gentler than REPETITION_PENALTY: those words are
# legitimately reusable across turns, and the goal is to break verbatim loops rather than to ban
# ordinary vocabulary. Only the last ECHO_HISTORY_TURNS replies are penalised, so something said
# a while ago is fair game again.
ECHO_PENALTY = 1.15
ECHO_HISTORY_TURNS = 2
ANSWER_TOKEN_RESERVE = MAX_NEW_TOKENS + 16

# Training data is full of these labels, so left alone the model keeps writing the
# document and invents its own next turn.
STOP_SEQUENCES = ("\nUser:", "\nContext:", "\nAssistant:")

# Below this the router is guessing. Measured: 0.6 routes ~83% of requests with
# ~89% of those correct.
ROUTER_CONFIDENCE = 0.6

# Retrieval is the default. These three are the categories the router separates
# reliably (f1: maths .995, summarise .917, chitchat .780) and none of them need a
# lookup. The factual/instruction boundary is NOT reliable (.678/.706), so anything
# else gets a passage rather than being trusted to answer from memory.
NO_RETRIEVAL_CATEGORIES = {"chitchat", "maths", "summarise"}

# No maths model exists yet - these fall through to the general model.
EXPERT_CATEGORIES = {"maths"}

# Conversation modes - the "knob" between looking things up and just talking.
#
#   auto  the router decides per message (default, and what everything before 2026-08-19 did)
#   chat  never retrieve; the whole context budget goes to conversation history
#   ask   always retrieve, except for questions about the conversation itself
#
# Why chat mode is worth having rather than just trusting the router: retrieval is not free. The
# context is 768 tokens with 96 reserved for the answer, leaving 672 - and measured, passages eat
# 85 to 311 of those (up to 46%). Every token a passage takes is a token of conversation the model
# cannot see, so on a purely conversational exchange retrieval actively costs memory while adding
# nothing. Chat mode roughly doubles how much conversation survives.
MODES = ("auto", "chat", "ask")
DEFAULT_MODE = "auto"

# Drop conversation history from the prompt whenever retrieval fired. See the comment in
# Assistant.stream_answer for the measurement: four factual questions x 5 samples went from
# 7/20 correct with history to 20/20 without it. Chat turns set a conversational register and
# the model answers in that register instead of reading the retrieved passage.
DROP_HISTORY_WHEN_RETRIEVING = True
