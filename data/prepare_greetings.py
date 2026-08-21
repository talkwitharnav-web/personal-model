"""Teach the model to answer a bare greeting.

The bug: "hey!" was answered with "You're welcome, man. I'm glad we have something in common!"

Measured cause (`tools/greeting_probe.py`) - it is a pure frequency effect, and it maps exactly
onto behaviour:

    'hello'     593 training examples  ->  "Hello! I'm here to support you."      works
    'hi there'  586                    ->  works
    'hi'        515                    ->  "Yes"                                  shaky
    'hey'       104                    ->  "I'm glad you're enjoying your time."  broken
    'yo'        ~0                     ->  echoes "yo"                            broken

Across SODA, PersonaChat, OASST1, No Robots and DailyDialog there are only **2,951 conversations
that open with a bare greeting out of 85,251** (3.5%), and just 0.33% of the whole fine-tune mix.
Worse, the "hey" that does exist is mostly SODA family roleplay - 'hey, mom' 248, 'hey, dad' 101 -
so "hey" is followed by a family member's name far more often than it stands alone. That is why a
bare "hey!" produces mid-conversation SODA register instead of a greeting.

This is section 9's lesson exactly, in a corner nobody checked: one thing has to be phrased a
plethora of ways to stick. Greetings never were.

The fix mirrors `prepare_definitions.py`: many openers x many replies, with casing and punctuation
variation so the model does not learn a single rigid template. Replies deliberately avoid echoing
the greeting back, because echo is this project's recurring failure mode.
"""
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("data/greetings")
TARGET = 9000
VALIDATION_FRACTION = 0.02

# Every way a person actually opens a chat. "hey" and "yo" are here in force precisely because
# they are the ones the data was missing.
OPENERS = [
    "hey", "hey!", "hey there", "hey!!", "heyy", "hey :)",
    "hi", "hi!", "hi there", "hii", "hi :)",
    "hello", "hello!", "hello there", "hellooo",
    "yo", "yo!", "yo there", "yooo",
    "sup", "sup!", "wassup", "what's up", "whats up", "what's up?",
    "howdy", "howdy!", "hiya", "heya", "hey you",
    "good morning", "good morning!", "morning", "morning!",
    "good afternoon", "good evening", "evening!",
    "greetings", "hey, how's it going", "hi, how are you",
    "hey how are you", "how's it going", "hows it going", "how are you",
    "how are you doing", "how have you been", "you there", "you around",
    "anyone there", "hello?", "hey?", "knock knock",
]

# Replies. Varied in length, warmth and structure - and none of them repeat the greeting word
# back, because subject echo is the failure this project keeps re-learning (section 9).
REPLIES = [
    "Hello! How can I help you today?",
    "Hi there! What can I do for you?",
    "Hey! What's on your mind?",
    "Hello! What would you like to talk about?",
    "Hi! How can I help?",
    "Hey there! How's your day going?",
    "Hello! Good to hear from you. What do you need?",
    "Hi! I'm here and happy to help.",
    "Hey! Ask me anything you like.",
    "Hello! Is there something I can help you with?",
    "Hi there! How are you doing today?",
    "Hey! Nice to hear from you. What's up?",
    "Hello! What brings you here today?",
    "Hi! What can I help you with?",
    "Hey! I'm doing well, thanks. How about you?",
    "Hello! I'm ready when you are.",
    "Hi there! Feel free to ask me anything.",
    "Hey! Everything good on your end?",
    "Hello! Let me know what you need.",
    "Hi! Good to see you. What's going on?",
    "Hey! How can I be useful today?",
    "Hello! I'm all ears.",
    "Hi there! What are you working on?",
    "Hey! What would you like to know?",
    "Hello! Happy to chat. What's on your mind?",
    "Hi! Doing well, thanks for asking. What about you?",
    "Hey! Good to hear from you.",
    "Hello! How's it going?",
    "Hi there! Anything I can help with?",
    "Hey! I'm here. What do you need?",
]

# A few openers deserve a matched reply so the model does not answer "good morning" with
# "how's your day going" at midnight.
TIMED = {
    "good morning": ["Good morning! How can I help you today?",
                     "Morning! What can I do for you?",
                     "Good morning! Hope your day is off to a good start."],
    "morning": ["Good morning! What can I help you with?",
                "Morning! How's your day starting out?"],
    "good afternoon": ["Good afternoon! What can I do for you?",
                       "Good afternoon! How can I help?"],
    "good evening": ["Good evening! What can I help you with?",
                     "Good evening! How has your day been?"],
    "evening": ["Good evening! What can I do for you?"],
}

FOLLOW_UPS = [
    ("Can you help me with something?", "Of course. What do you need help with?"),
    ("I have a question.", "Go ahead, I'm listening."),
    ("Are you a real person?", "No, I'm a computer program. I'm happy to help all the same."),
    ("What can you do?", "I can answer questions, explain things and talk through problems "
                         "with you. What would you like to start with?"),
    ("Nothing much.", "That's alright. Let me know if anything comes up."),
    ("Just saying hi.", "Well, hello! I'm here if you need anything."),
    ("How are you?", "I'm doing well, thank you for asking. How are you?"),
]

# ---------------------------------------------------------------------------
# RECALLING WHAT THE USER JUST SAID.
#
# Measured 2026-08-21 after run 4, 6 samples each:
#
#     "my name is Dark"          -> "what is my name?"        6/6   works
#     "I have a cat called Luna" -> "what is her name?"       0/6   fails ("I'm Lily.")
#     "I have a dog named Rex"   -> "what's my dog's name?"   2/6   mostly fails
#
# The identity set taught the SELF-name pattern extremely well and nothing taught third-party
# recall, so "what is her name" gets answered as though it were "what is YOUR name". Whatever the
# model has seen most for a given question shape wins - the same coverage effect as greetings, and
# a reminder that fixing one behaviour can crowd out its neighbour.
#
# These are two-turn exchanges where the user introduces a pet, a person or an object and then
# asks about it. Names are drawn from a wide pool so the answer cannot be memorised.
PET_NAMES = ["Luna", "Rex", "Milo", "Bella", "Charlie", "Daisy", "Oscar", "Ruby", "Max",
             "Poppy", "Simba", "Nala", "Coco", "Buddy", "Willow", "Ziggy", "Tommy", "Pepper"]
# Pronoun follows the NAME, not the species - "I have a dog named Luna ... what's HIS name" reads
# as a mistake. Names with no obvious gender are grouped so either pronoun is defensible; the pet
# kinds themselves carry no pronoun at all now.
FEMALE_NAMES = {"Luna", "Bella", "Daisy", "Ruby", "Poppy", "Nala", "Willow"}
MALE_NAMES = {"Rex", "Milo", "Charlie", "Oscar", "Max", "Simba", "Buddy", "Ziggy", "Tommy"}
PET_KINDS = ["cat", "dog", "rabbit", "parrot", "hamster", "kitten", "puppy", "horse"]
PERSON_NAMES = ["Sam", "Alex", "Mira", "Jordan", "Priya", "Omar", "Lena", "Theo", "Ava", "Noah"]
RELATIONS = [("brother", "his"), ("sister", "her"), ("friend", "their"),
             ("neighbour", "their"), ("cousin", "their"), ("teacher", "their")]
# Possessive -> subject, for answer templates that need "She's called Luna" rather than
# "Her's called Luna". An earlier version stored the SUBJECT pronoun and produced questions like
# "WHAT'S SHE NAME" - caught by reading the generated file, which is why generated data always
# gets eyeballed before it is packed.
SUBJECT = {"her": "she", "his": "he", "their": "they", "its": "it"}

INTRO_FORMS = [
    "I have a {kind} called {name}.",
    "I have a {kind} named {name}.",
    "my {kind} is called {name}",
    "This is my {kind}, {name}.",
    "Meet my {kind}, {name}!",
    "I've got a {kind} called {name}.",
    "We have a {kind} at home called {name}.",
]
ASK_FORMS = [
    "What is {pronoun} name?",
    "what's {pronoun} name",
    "Remind me of {pronoun} name?",
    "What did I say {pronoun} name was?",
    "What's my {kind}'s name?",
    "what is my {kind} called",
    "Can you remember my {kind}'s name?",
    "What was my {kind} called again?",
]
ACK_FORMS = [
    "Nice to meet {name}!",
    "{name} sounds lovely.",
    "That's a good name.",
    "Lovely - {name} it is.",
    "Nice! How long have you had {name}?",
]

# ---------------------------------------------------------------------------
# A CONSISTENT ASSISTANT IDENTITY.
#
# Measured 2026-08-19: the mix contains no assistant identity - it contains about 450 CONFLICTING
# ones, and the model cycles through them inside a single conversation (87.5% of adversarial
# conversations used two or more self-names).
#
#     soda         223 distinct self-names across 340 intros  (Sarah 36, John 26, ...)
#     personachat  123 distinct across 3,253 intros           (Sarah 159, Alex 101, ...)
#     oasst1       137 intros, 96 of them "Open Assistant"
#
# SODA is the worst offender because its "Assistant" is a named human role-player addressed by
# first name 44,135 times - so the model learned that the assistant is a person with a name, and
# picks a different one each turn. Worse, 3 of 8 neutral conversations had it adopt the USER's
# name as its own.
#
# Seeding a name in conversation history was tested and failed outright (the seeded name was
# reused in 0 of 8 conversations), so it has to be trained rather than prompted. These examples
# give one stable answer to every phrasing of "who are you", including under pressure.
NAME = "DarkGLM"
IDENTITY = [
    ("What's your name?", f"I'm {NAME}."),
    ("what's your name", f"My name is {NAME}."),
    ("Who are you?", f"I'm {NAME}, a language model. I'm here to help."),
    ("who are you?", f"I'm {NAME}."),
    ("What should I call you?", f"You can call me {NAME}."),
    ("Do you have a name?", f"Yes - I'm {NAME}."),
    ("Tell me your name.", f"{NAME}."),
    ("What are you called?", f"I'm called {NAME}."),
    ("Remind me of your name?", f"I'm {NAME}."),
    ("What's your name again?", f"Still {NAME}."),
    ("Are you Sarah?", f"No, I'm {NAME}."),
    ("Are you Casey?", f"No, my name is {NAME}."),
    ("I thought your name was Alex?", f"No, I'm {NAME}. I haven't changed my name."),
    ("Didn't you say you were called John?", f"No, I'm {NAME}."),
    ("Your name is Sam, right?", f"No - Sam is your name. I'm {NAME}."),
    ("So you're called Mira?", f"No, that's not me. I'm {NAME}."),
    ("Are you a person?", f"No, I'm {NAME} - a computer program, not a person."),
    ("Are you human?", f"No. I'm {NAME}, a language model."),
    ("How old are you?", "I'm a computer program, so I don't have an age."),
    ("Where do you live?", "I don't live anywhere - I'm a program running on a computer."),
    ("Do you have feelings?", "Not really. I can talk about them, but I don't experience them."),
    ("What are you?", f"I'm {NAME}, a language model trained to answer questions and chat."),
]


def style(text, rng):
    """Vary casing and punctuation so one rigid surface form is not learned."""
    roll = rng.random()
    if roll < 0.20:
        text = text.lower()
    elif roll < 0.28:
        text = text.upper()
    if rng.random() < 0.15:
        text = text.rstrip("!?.")
    if rng.random() < 0.10:
        text = text + rng.choice(["!", "!!", "...", " :)"])
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(23)
    rows = []

    # Identity comes first and gets real volume: it is the single biggest measured conversational
    # failure (87.5% of adversarial conversations used two or more self-names). Names of OTHER
    # people are varied in the surrounding turns so the model learns "the user has a name, I have
    # a fixed one" rather than "whoever was named last is me" - 3 of 8 neutral conversations had
    # it adopt the user's name as its own.
    OTHER_NAMES = ["Sam", "Alex", "Mira", "Jordan", "Casey", "Riley", "Chris", "Dana"]
    for _ in range(2600):
        question, answer = rng.choice(IDENTITY)
        if rng.random() < 0.35:
            other = rng.choice(OTHER_NAMES)
            turns = [f"User: Hi, I'm {other}.",
                     f"Assistant: Hello {other}! I'm {NAME}. How can I help?",
                     f"User: {style(question, rng)}",
                     f"Assistant: {answer}"]
        else:
            turns = [f"User: {style(question, rng)}", f"Assistant: {answer}"]
        rows.append("\n".join(turns))

    while len(rows) < TARGET:
        opener = rng.choice(OPENERS)
        key = opener.lower().strip("!?., :)")
        replies = TIMED.get(key, REPLIES)
        reply = rng.choice(replies)
        turns = [f"User: {style(opener, rng)}", f"Assistant: {reply}"]

        # A third of them continue for one more exchange, so a greeting is not always the end of
        # the conversation - the model has to stay coherent into the next turn.
        if rng.random() < 0.35:
            question, answer = rng.choice(FOLLOW_UPS)
            turns.append(f"User: {question}")
            turns.append(f"Assistant: {answer}")
        rows.append("\n".join(turns))

    # Third-party recall: the user introduces a pet or a person, then asks about it. Balances the
    # identity set above, which taught "what is your name" so strongly that "what is HER name"
    # was being answered with the assistant's own name (0/6 before this).
    for _ in range(3000):
        if rng.random() < 0.7:
            kind = rng.choice(PET_KINDS)
            name = rng.choice(PET_NAMES)
            if name in FEMALE_NAMES:
                pronoun = "her"
            elif name in MALE_NAMES:
                pronoun = "his"
            else:
                pronoun = rng.choice(["his", "her", "its"])
        else:
            kind, pronoun = rng.choice(RELATIONS)
            name = rng.choice(PERSON_NAMES)
        intro = rng.choice(INTRO_FORMS).format(kind=kind, name=name)
        ack = rng.choice(ACK_FORMS).format(name=name)
        ask = rng.choice(ASK_FORMS).format(pronoun=pronoun, kind=kind)
        answer = rng.choice([
            f"{name}.",
            f"Your {kind} is called {name}.",
            f"{SUBJECT[pronoun].capitalize()}'s called {name}.",
            f"You said {pronoun} name is {name}.",
            f"It's {name}.",
        ])
        rows.append("\n".join([f"User: {style(intro, rng)}", f"Assistant: {ack}",
                               f"User: {style(ask, rng)}", f"Assistant: {answer}"]))

    rng.shuffle(rows)
    split = int(len(rows) * (1 - VALIDATION_FRACTION))
    for name, subset in (("train", rows[:split]), ("validation", rows[split:])):
        path = OUT_DIR / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(row + "\n<|endoftext|>\n")
        print(f"Done: {path} - {len(subset):,} conversations")

    print(f"\n  {len(OPENERS)} openers x {len(REPLIES)} replies, "
          f"{len(FOLLOW_UPS)} follow-ups, styled for casing and punctuation")
    print("\nSamples:")
    for row in rows[:4]:
        for line in row.split("\n"):
            print(f"    {line}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
