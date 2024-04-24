# Beam Search

This is a python-based example implementation of beam search, leveraging Hugging Face's `transformers` package to use a `T5Tokenizer` and `T5`-based `Model`

Example usage:

```
  $ python3 src/main.py --beam 4 --input ./wiki-fox.txt --prompt "summarize the following document"

  [ voluminous output ]

  beam search (k=4, t=4.0) generation results:
  [
   "the quick brown fox jumps over the lazy dog" is an English-language pangram. 
   it is commonly used for touch-typing practice, testing typewriters and 
   computer keyboards. earliest known use of the phrase started with "A"
  ]
```
