# Beam Search

This is a python-based example implementation of beam search, leveraging Hugging Face's `transformers` package to use a `T5Tokenizer` and `T5`-based `Model`.

This code and algorithm is discussed in detail in this blog post: [Temperature Scaling and Beam Search Generation in LLMs, for the ML-Adjacent](https://towardsdatascience.com/temperature-scaling-and-beam-search-text-generation-in-llms-for-the-ml-adjacent-21212cc5dddb)

Example usage:

```
  $ python3 src/main.py --beam 4 --temperature 4.0 --input ./corpora/wiki-fox.txt --prompt "summarize the following document"

  [ voluminous output ]

  beam search (k=4, t=4.0) generation results:
  [
   "the quick brown fox jumps over the lazy dog" is an English-language pangram. 
   it is commonly used for touch-typing practice, testing typewriters and 
   computer keyboards. earliest known use of the phrase started with "A"
  ]
```
