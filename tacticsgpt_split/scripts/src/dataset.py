import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


def split_article_documents(text: str) -> list[str]:
    docs = [part.strip() for part in re.split(r"\n{3,}", text) if part.strip()]
    return docs if docs else [text.strip()]


class TacticsDataset(Dataset):
    def __init__(
        self,
        corpus_path: str,
        tokenizer_path: str,
        context_length: int = 256,
        stride: int | None = None,
    ) -> None:
        corpus = Path(corpus_path)
        if not corpus.exists() or corpus.stat().st_size == 0:
            raise FileNotFoundError(f"Corpus not found or empty: {corpus}")

        self.context_length = context_length
        self.stride = stride or context_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.documents: list[list[int]] = []
        self.examples: list[tuple[int, int]] = []

        text = corpus.read_text(encoding="utf-8", errors="ignore")
        for document in split_article_documents(text):
            token_ids = self.tokenizer.encode(document).ids
            if len(token_ids) < context_length + 1:
                continue
            doc_index = len(self.documents)
            self.documents.append(token_ids)
            self.examples.extend(
                (doc_index, start)
                for start in range(0, len(token_ids) - context_length, self.stride)
            )

        self.num_documents = len(self.documents)
        self.num_tokens = sum(len(doc) for doc in self.documents)
        if not self.examples:
            raise ValueError(
                "No training windows were created. "
                f"Try reducing context_length below {context_length}, "
                "or check that data/tactics_corpus.txt contains full article bodies."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        doc_index, start = self.examples[idx]
        tokens = self.documents[doc_index]
        end = start + self.context_length
        x = torch.tensor(tokens[start:end], dtype=torch.long)
        y = torch.tensor(tokens[start + 1 : end + 1], dtype=torch.long)
        return x, y
