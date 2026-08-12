from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel()

special_tokens = ["<unk>", "<pad>", "<|endoftext|>"]

trainer = BpeTrainer(
    vocab_size=32000,
    special_tokens=special_tokens,
    min_frequency=2,
)

files = ["data/tinystories/train.txt", "data/wikitext_sample.txt"]
tokenizer.train(files, trainer)

tokenizer.save("tokenizer/tokenizer.json")
print("Tokenizer trained and saved to tokenizer/tokenizer.json (that's the path)")

test_tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
encoded = test_tokenizer.encode("Once upon a time, there was a little car named Beep.")
print(encoded.tokens)
print(encoded.ids)