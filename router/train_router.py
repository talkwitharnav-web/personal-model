# Trains the request-type router: MiniLM turns each question into 384 numbers
# (frozen, never trained), a logistic regression learns the category boundaries.
import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from router.build_router import build_dataset, CATEGORIES

OUTPUT_PATH = Path("router/router.joblib")

HELD_OUT_CHECKS = [
    ("what is 45 times 12", "maths"),
    ("if i buy 3 shirts at 14 dollars each how much do i spend", "maths"),
    ("give me the gist of this article: the council announced on tuesday that", "summarise"),
    ("who was the first president of the united states", "factual"),
    ("can i put a metal fork in the microwave", "commonsense"),
    ("all cats are animals and fluffy is a cat, so is fluffy an animal", "reasoning"),
    ("hey how is your day going", "chitchat"),
    ("write me a poem about the sea", "instruction"),
]

questions, labels = build_dataset()
print(f"\n{len(questions):,} questions across {len(CATEGORIES)} categories")

train_q, test_q, train_y, test_y = train_test_split(
    questions, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(train_q):,}  Test: {len(test_q):,}")

print("\nLoading MiniLM (frozen - we never train it)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

print("Embedding questions...")
train_x = embedder.encode(train_q, batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=True)
test_x = embedder.encode(test_q, batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)
print(f"Embeddings: {train_x.shape} (one row per question, 384 numbers each)")

print("\nTraining the classifier...")
classifier = LogisticRegression(max_iter=2000, C=10.0)
classifier.fit(train_x, train_y)

train_accuracy = classifier.score(train_x, train_y)
test_accuracy = classifier.score(test_x, test_y)
print(f"Train accuracy: {train_accuracy:.3f}")
print(f"Test accuracy:  {test_accuracy:.3f}")

print("\n--- per category (held-out rows) ---")
print(classification_report(test_y, classifier.predict(test_x), digits=3))

print("--- confusion matrix (rows = true, columns = predicted) ---")
names = list(classifier.classes_)
matrix = confusion_matrix(test_y, classifier.predict(test_x), labels=names)
print(f"{'':14}" + "".join(f"{n[:9]:>10}" for n in names))
for name, row in zip(names, matrix):
    print(f"{name:14}" + "".join(f"{value:>10}" for value in row))

# The real test: questions written by hand, not lifted from any training file.
print("\n--- hand-written questions (nothing like the training rows) ---")
check_x = embedder.encode([q for q, _ in HELD_OUT_CHECKS], normalize_embeddings=True,
                          convert_to_numpy=True)
predictions = classifier.predict(check_x)
probabilities = classifier.predict_proba(check_x)
correct = 0
for (question, expected), predicted, probability in zip(HELD_OUT_CHECKS, predictions, probabilities):
    confidence = probability[list(classifier.classes_).index(predicted)]
    mark = "ok " if predicted == expected else "MISS"
    correct += predicted == expected
    print(f"{mark} {predicted:12} {confidence:.2f}  (expected {expected:12}) {question}")
print(f"{correct}/{len(HELD_OUT_CHECKS)} correct")

# Below a confidence threshold it's better to fall back to the general model
# than to send the request to the wrong specialist.
print("\n--- confidence threshold tradeoff ---")
test_probabilities = classifier.predict_proba(test_x)
test_predictions = classifier.predict(test_x)
best = test_probabilities.max(axis=1)
hit = test_predictions == np.array(test_y)
for threshold in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
    confident = best >= threshold
    routed = confident.mean()
    accuracy = hit[confident].mean() if confident.any() else 0.0
    print(f"  threshold {threshold:.1f}: routes {routed:6.1%} of requests, {accuracy:.1%} of those correct")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(classifier, OUTPUT_PATH)
print(f"\nSaved classifier to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
