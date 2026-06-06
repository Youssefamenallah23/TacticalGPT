import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_tokenizer(corpus_path: str, out_dir: str, vocab_size: int = 8000, lowercase: bool = False) -> None:
    corpus = Path(corpus_path)
    if not corpus.exists() or corpus.stat().st_size == 0:
        raise FileNotFoundError(f"Corpus not found or empty: {corpus}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    normalizers = [NFKC()]
    if lowercase:
        normalizers.append(Lowercase())
    tokenizer.normalizer = Sequence(normalizers)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train([str(corpus)], trainer)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    tokenizer.save(str(out / "tokenizer.json"))
    tokenizer.model.save(str(out))

    print("Saved tokenizer files:")
    print(" -", out / "tokenizer.json")
    print(" -", out / "vocab.json")
    print(" -", out / "merges.txt")
    print("Actual vocab size:", tokenizer.get_vocab_size())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/tactics_corpus.txt")
    parser.add_argument("--out_dir", default="checkpoints/tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()
    train_tokenizer(args.corpus, args.out_dir, args.vocab_size, args.lowercase)


if __name__ == "__main__":
    main()
