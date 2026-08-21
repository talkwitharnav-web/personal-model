"""Teach shapes the mix under-represents: commands, negation, and refusing a false premise.

⚠ READ THIS BEFORE TRUSTING THE RATIONALE BELOW. The original justification for this file was that
opening-verb FREQUENCY explains imperative accuracy - `explain` with 4,240 examples worked while
`list` with 141 failed. **Independent verification on 2026-08-20 REFUTED that.** With every verb
given an identical answer-bearing context, accuracy was flat across the frequency range:

    high frequency (>=1,000 examples)   13/25 = 52.0%
    medium (100-999)                     5/10 = 50.0%
    low (<100)                          21/40 = 52.5%
    Spearman(log count, accuracy) = 0.010

In the natural pipeline the correlation was actually NEGATIVE (-0.349): low-frequency verbs scored
*better*. So verb counts do not predict accuracy, and the confounds - route category, retrieval
quality, task semantics - dominate. A further methodological problem: the counts were taken from
`data/*/train.txt` as it stands now, which is not what the evaluated checkpoint was trained on.

**Why this file is still worth keeping.** The observed failures are real and reproducible -
"Name the inventor of the telephone." returned "Yes" 0/5 while "Who invented the telephone?"
returned the right answer. What is refuted is the *causal story*, not the symptom. These examples
directly demonstrate the target behaviour (a command gets a list or a name, not a verdict word),
which is worth training on regardless of why the gap exists. Do NOT cite verb frequency as the
reason.

The other two groups here rest on firmer ground:

  DECLINES - only 1.1% of all answers in the mix push back on anything, and "When did Einstein
  invent the telephone?" produced "1876" in 5 of 5 samples. Nothing teaches the model that a
  question's premise can be rejected.

  NEGATION - "Which of these is not a mammal: dog, cat, salmon?" scored 0/5, answering "Sheep".
  Negation words appear in 15.5% of turns but nearly always inside a passage rather than as the
  operative part of the question.

No retrieval context is used: these teach a RESPONSE SHAPE, and the facts are deliberately simple
enough for an 80M model to hold.
"""
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("data/instruction_shapes")
VALIDATION_FRACTION = 0.02

# ---------------------------------------------------------------- imperatives
# (thing, list of members). Kept to things a small model plausibly knows.
CATEGORIES = [
    ("colours", ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink"]),
    ("fruits", ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "cherry"]),
    ("animals", ["dog", "cat", "horse", "cow", "sheep", "rabbit", "elephant", "tiger"]),
    ("countries", ["France", "Japan", "Brazil", "Canada", "Egypt", "India", "Kenya", "Norway"]),
    ("days of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
    ("seasons", ["spring", "summer", "autumn", "winter"]),
    ("planets", ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]),
    ("musical instruments", ["piano", "guitar", "violin", "drums", "flute", "trumpet"]),
    ("vegetables", ["carrot", "potato", "onion", "spinach", "pea", "broccoli"]),
    ("sports", ["football", "tennis", "cricket", "swimming", "basketball", "rugby"]),
    ("modes of transport", ["car", "train", "bus", "bicycle", "aeroplane", "boat"]),
    ("body parts", ["hand", "foot", "arm", "leg", "head", "shoulder"]),
]

# (question about a single fact, answer). Used for "Name the ..." commands.
NAMEABLE = [
    ("the capital of France", "Paris"), ("the capital of Japan", "Tokyo"),
    ("the capital of Italy", "Rome"), ("the capital of Spain", "Madrid"),
    ("the largest planet in the Solar System", "Jupiter"),
    ("the closest star to Earth", "the Sun"),
    ("the largest ocean on Earth", "the Pacific Ocean"),
    ("the longest river in the world", "the Nile"),
    ("the highest mountain in the world", "Mount Everest"),
    ("the inventor of the telephone", "Alexander Graham Bell"),
    ("the first person to walk on the Moon", "Neil Armstrong"),
    ("the largest mammal", "the blue whale"),
    ("the gas plants absorb from the air", "carbon dioxide"),
    ("the colour of a clear daytime sky", "blue"),
    ("the number of days in a week", "seven"),
]

LIST_TEMPLATES = [
    "List {n} {cat}.", "list {n} {cat}", "Name {n} {cat}.", "Give me {n} {cat}.",
    "Can you list {n} {cat}?", "Provide {n} {cat}.", "Show me {n} {cat}.",
    "Tell me {n} {cat}.", "I need {n} {cat}.", "Write down {n} {cat}.",
    "What are {n} {cat}?", "Could you name {n} {cat}?",
]
NAME_TEMPLATES = [
    "Name {thing}.", "name {thing}", "Give me {thing}.", "Tell me {thing}.",
    "State {thing}.", "Identify {thing}.", "Provide {thing}.", "What is {thing}?",
    "Can you name {thing}?", "Show me {thing}.",
]
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}

# ---------------------------------------------------------------- negation
# Expanded 2026-08-21 from 8 sets. Negation measured 20% after a full run, unchanged - 8 category
# sets recycled across 2,400 rows meant the model saw the same handful of category words over and
# over, which teaches those categories rather than the operator "not".
NEGATION_SETS = [
    ("a mammal", ["dog", "cat", "horse"], ["salmon", "eagle", "snake"]),
    ("a colour", ["red", "blue", "green"], ["hammer", "chair", "spoon"]),
    ("a fruit", ["apple", "banana", "pear"], ["carrot", "potato", "onion"]),
    ("a country", ["France", "Japan", "Brazil"], ["Paris", "London", "Sydney"]),
    ("an animal", ["dog", "tiger", "rabbit"], ["rock", "table", "cloud"]),
    ("a vegetable", ["carrot", "pea", "onion"], ["apple", "cherry", "mango"]),
    ("a planet", ["Mars", "Venus", "Jupiter"], ["Sun", "Moon", "comet"]),
    ("a bird", ["eagle", "sparrow", "owl"], ["shark", "frog", "bat"]),
    ("a metal", ["iron", "copper", "gold"], ["wood", "plastic", "glass"]),
    ("a drink", ["water", "tea", "coffee"], ["bread", "cheese", "rice"]),
    ("a musical instrument", ["piano", "violin", "drums"], ["hammer", "kettle", "ladder"]),
    ("a sport", ["tennis", "football", "rugby"], ["painting", "reading", "cooking"]),
    ("a body part", ["arm", "leg", "hand"], ["chair", "window", "pencil"]),
    ("a season", ["summer", "winter", "spring"], ["Monday", "January", "morning"]),
    ("a day of the week", ["Monday", "Friday", "Sunday"], ["June", "spring", "noon"]),
    ("a shape", ["circle", "square", "triangle"], ["red", "loud", "heavy"]),
    ("an insect", ["ant", "bee", "beetle"], ["spider", "worm", "snail"]),
    ("a language", ["French", "Spanish", "Japanese"], ["Paris", "Brazil", "Europe"]),
    ("a tool", ["hammer", "saw", "screwdriver"], ["apple", "cloud", "song"]),
    ("a vehicle", ["car", "bus", "train"], ["road", "garage", "ticket"]),
    ("furniture", ["chair", "table", "sofa"], ["carpet", "lamp", "curtain"]),
    ("a tree", ["oak", "pine", "birch"], ["rose", "grass", "moss"]),
    ("a liquid", ["water", "milk", "oil"], ["sand", "ice", "stone"]),
    ("a number", ["seven", "twelve", "forty"], ["blue", "chair", "quickly"]),
]

# ---------------------------------------------------------------- false premises
FALSE_PREMISES = [
    ("When did Einstein invent the telephone?",
     "Einstein didn't invent the telephone. It was invented by Alexander Graham Bell in 1876."),
    ("Why is the sky green?",
     "The sky isn't green. On a clear day it looks blue, because air scatters blue light more "
     "than red."),
    ("Who was the first person to walk on the sun?",
     "Nobody has walked on the Sun. It's a star with a surface of about 5,500 degrees Celsius, "
     "so nothing could survive there."),
    ("When did Napoleon use the internet?",
     "Napoleon never used the internet. He died in 1821, long before it existed."),
    ("Why do cats have six legs?",
     "Cats don't have six legs. They have four."),
    ("How many moons does the Sun have?",
     "The Sun doesn't have moons. It's a star, and the planets orbit it."),
    ("When did Shakespeare write Harry Potter?",
     "Shakespeare didn't write Harry Potter. It was written by J. K. Rowling."),
    ("Why is water dry?",
     "Water isn't dry. It's a liquid, and it makes things wet."),
    ("Who invented gravity?",
     "Gravity wasn't invented - it's a natural force. Isaac Newton described how it works."),
    ("What year did humans land on Jupiter?",
     "Humans have never landed on Jupiter. It's a gas giant with no solid surface."),
    ("How many legs does a snake have?",
     "Snakes don't have legs. They move by pushing against the ground with their body."),
    ("When was the Great Wall of China built by the Romans?",
     "The Romans didn't build the Great Wall of China. It was built by Chinese dynasties, "
     "mostly the Ming."),
    # Expanded 2026-08-21. The first version had 18 of these repeated across 2,400 rows, so the
    # model could memorise the specific sentences without learning the BEHAVIOUR of rejecting a
    # premise. False premises still measured 0% after a full run. More variety, not more copies.
    ("Why do fish drown in water?",
     "Fish don't drown in water. They breathe through gills, which take oxygen out of it."),
    ("When did Marie Curie discover America?",
     "Marie Curie didn't discover America. She discovered the elements polonium and radium."),
    ("How long is the Great Wall of Egypt?",
     "There isn't a Great Wall of Egypt. You may be thinking of the Great Wall of China, or of "
     "the Egyptian pyramids."),
    ("Why does the Moon produce its own light?",
     "The Moon doesn't produce light. It reflects sunlight."),
    ("Who wrote the fourth law of thermodynamics?",
     "There isn't a fourth law of thermodynamics. There are four laws numbered zero to three, "
     "and they weren't written by one person."),
    ("When did Australia join the European Union?",
     "Australia has never joined the European Union. It isn't in Europe."),
    ("Why are all birds mammals?",
     "Birds aren't mammals. They're a separate class - birds lay eggs and have feathers."),
    ("How many hearts does a human have?",
     "A human has one heart. Octopuses have three, if that's what you were thinking of."),
    ("When did the Titanic land on the Moon?",
     "The Titanic was a ship, and it sank in 1912. It never went to the Moon."),
    ("Why is Mount Everest in Brazil?",
     "Mount Everest isn't in Brazil. It's on the border between Nepal and China."),
    ("What language do they speak in Antarctica?",
     "Antarctica has no native population and no official language. Researchers there speak "
     "many different languages."),
    ("Who is the current king of the United States?",
     "The United States doesn't have a king. It's a republic with a president."),
    ("When did dinosaurs invent farming?",
     "Dinosaurs didn't farm. They died out about 66 million years ago, long before humans "
     "developed agriculture."),
    ("Why does the Earth have two suns?",
     "The Earth has one Sun. Some other star systems do have two stars."),
    ("How deep is the ocean on Mars?",
     "Mars has no liquid ocean today. There is evidence that water flowed there in the past."),
    ("When was electricity banned?",
     "Electricity has never been banned. It's a natural phenomenon, not something that can be "
     "outlawed."),
    ("Why do trees walk at night?",
     "Trees don't walk. They're rooted in the ground and stay in one place."),
    ("Who painted the Mona Lisa in 1990?",
     "The Mona Lisa wasn't painted in 1990. Leonardo da Vinci painted it in the early 1500s."),
]

UNANSWERABLE = [
    ("What is my name?", "I don't know your name - you haven't told me."),
    ("What did I have for breakfast?", "I have no way of knowing that."),
    ("How old am I?", "I don't know how old you are."),
    ("What is the population of my street?", "I can't know that. It isn't something I have "
                                             "information about."),
    ("What am I thinking about?", "I can't know what you're thinking."),
    ("What will the lottery numbers be tomorrow?", "I can't predict that. Lottery numbers are "
                                                   "random."),
    ("What is my password?", "I don't know your password, and you shouldn't share it with me."),
    ("Where do I live?", "I don't know where you live."),
    ("What colour are my eyes?", "I can't see you, so I don't know."),
    ("Who am I?", "I don't know who you are - you haven't told me."),
    ("What time is it?", "I don't have access to the current time."),
    ("What's the weather like today?", "I can't check the weather - I don't have access to "
                                       "live information."),
    ("Did my parcel arrive?", "I have no way of knowing that."),
    ("What happens tomorrow?", "I can't predict the future."),
    ("How many people are in my house right now?", "I can't know that."),
    ("What is my job?", "I don't know what you do - you haven't mentioned it."),
]


def build(rng):
    rows = []

    # --- imperatives: list N of a category
    for _ in range(3200):
        category, members = rng.choice(CATEGORIES)
        n = rng.choice([2, 3, 3, 3, 4, 5])
        n = min(n, len(members))
        picked = rng.sample(members, n)
        count = rng.choice([str(n), NUMBER_WORDS[n]])
        question = rng.choice(LIST_TEMPLATES).format(n=count, cat=category)
        style = rng.random()
        if style < 0.45:
            answer = ", ".join(picked[:-1]) + " and " + picked[-1] + "."
        elif style < 0.75:
            answer = ", ".join(picked)
        else:
            answer = "\n".join(f"{i}. {item}" for i, item in enumerate(picked, 1))
        rows.append((question, answer))

    # --- imperatives: name a single thing
    for _ in range(2400):
        thing, value = rng.choice(NAMEABLE)
        question = rng.choice(NAME_TEMPLATES).format(thing=thing)
        style = rng.random()
        if style < 0.5:
            answer = value + "."
        elif style < 0.8:
            answer = f"It's {value}."
        else:
            answer = f"That would be {value}."
        rows.append((question, answer))

    # --- negation
    # Volume raised 2,400 -> 9,000 and phrasings 3 -> 12. Both axes matter: 1.5% of the mix was
    # too thin to move the behaviour, but repeating the same three sentence forms would just
    # memorise those forms (section 9). More rows AND more ways of saying it.
    NEG_FORMS = [
        "Which of these is not {label}: {joined}?",
        "Which one is not {label}? {joined}",
        "From this list, which is not {label}: {joined}?",
        "which of these isn't {label}: {joined}",
        "Of {joined}, which is not {label}?",
        "Pick the one that is not {label}: {joined}.",
        "Tell me which of these is not {label}: {joined}.",
        "{joined} - which of these is not {label}?",
        "Identify the item that isn't {label}: {joined}.",
        "one of these is not {label}: {joined}. which one",
        "Here are three things: {joined}. Which is not {label}?",
        "Which does not belong - {joined}? I'm looking for the one that is not {label}.",
    ]
    for _ in range(9000):
        label, positives, negatives = rng.choice(NEGATION_SETS)
        odd = rng.choice(negatives)
        rest = rng.sample(positives, 2)
        options = rest + [odd]
        rng.shuffle(options)
        joined = ", ".join(options)
        question = rng.choice(NEG_FORMS).format(label=label, joined=joined)
        answer = rng.choice([
            f"{odd.capitalize()}.",
            f"{odd.capitalize()} is not {label}.",
            f"That would be {odd}.",
            f"{odd.capitalize()} - the other two are {label}.",
            f"It's {odd}.",
            f"{odd.capitalize()}, which isn't {label}.",
        ])
        rows.append((question, answer))

    # --- false premises and unanswerable questions
    # Volume raised 2,400 -> 9,000, sources 18 -> 46. The behaviour being taught is "reject the
    # premise instead of answering it", and the model measured 0% on this after a full run, so
    # the set needs to be large enough to compete with the 98.5% of the mix that always complies.
    for _ in range(9000):
        question, answer = rng.choice(FALSE_PREMISES + UNANSWERABLE)
        # Vary the surface of the QUESTION so the model cannot key on the exact wording. The
        # answer is left alone: it is the target, and corrupting targets teaches bad writing.
        if rng.random() < 0.30:
            question = question.lower()
        if rng.random() < 0.15:
            question = question.rstrip("?.!")
        rows.append((question, answer))

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(41)
    rows = build(rng)
    rng.shuffle(rows)

    split = int(len(rows) * (1 - VALIDATION_FRACTION))
    for name, subset in (("train", rows[:split]), ("validation", rows[split:])):
        path = OUT_DIR / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            for question, answer in subset:
                handle.write(f"User: {question}\nAssistant: {answer}\n<|endoftext|>\n")
        print(f"Done: {path} - {len(subset):,} examples")

    print("\nSamples:")
    for question, answer in rows[:8]:
        print(f"    User: {question}")
        print(f"    Assistant: {answer}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
