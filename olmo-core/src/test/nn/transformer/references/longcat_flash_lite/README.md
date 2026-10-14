# Released LongCat reference for row-selection tests

`modeling_longcat_ngram.py.txt` is an unmodified copy of
[Meituan's released implementation](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/blob/b62b68827ead0b7fef3ba98b57f18484acaaec06/modeling_longcat_ngram.py),
pinned to revision `b62b68827ead0b7fef3ba98b57f18484acaaec06`.
Its MIT [license](LICENSE) is copied from that same revision. The `.txt` suffix
keeps formatters from changing the reference and avoids importing the full model.

The source SHA-256 is
`d75c6bfc20a1c8045b3958eb2e2167d3e80bb110e8475d64390abe8eb160968f`.
The license SHA-256 is
`4722341d105cec173161f93ccab6d4b49409b368ece23faa22f2f74adddef9ed`.
Both files retain the upstream bytes, including whitespace.

`longcat_reference_test.py` verifies the source digest and executes its unchanged
`NgramEmbedding` class. It loads that class from the parsed source so the unrelated
Transformers backbone and its dependencies are not required. Tests call the
released history-shift helper and observe the row IDs passed to its embedding
tables during the released forward method. They compare those IDs with ours,
accounting for our concatenated table storage.

The comparison covers document boundaries and hash-based row selection. It does
not compare embedding outputs, gradients, initialization, or full-model behavior.
The tests run locally without downloading code, model weights, or configuration.
