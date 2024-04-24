import argparse
import beam
import greedy
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import torch

MODEL_ID = 't5-small'
TOKENIZER_ID = 'google-t5/t5-small'

def detect_device(device=None):
  """
  Depending on the provided arguments and the available device backends, returns either a CPU,
  or GPU device, where the GPU device may be either a CUDA-based GPU or Apple Silicon "MPS" based GPU.
  Default device is CPU.
  """
  if device == "cpu":
    return torch.device("cpu")
  
  if not device or device == "gpu":
    if torch.backends.mps.is_available():
      return torch.device("mps")
    elif torch.backends.cuda.is_available:
      return torch.device("cuda")

  return torch.device("cpu")  # Fallback to CPU if no GPU device is available

def run_greedy_search(tokenizer, model, input_ids, device):
  generated_ids = greedy.search(
    model, 
    input_ids, 
    max_length=64, 
    start_token_id=tokenizer.pad_token_id, 
    end_token_id=tokenizer.eos_token_id, 
    device=device
  )

  # Decode the token IDs back to strings
  decoded_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
  
  print(f"greedy search generation results:\n[\n{decoded_text}\n]")

def run_beam_search(tokenizer, model, input_ids, device, width, temperature=1.0, decay=False):
  beam_ids = beam.search(
    tokenizer, 
    model, 
    input_ids, 
    # T5 uses the pad_token_id as the starting token for 
    # text generation (https://huggingface.co/docs/transformers/main/en/model_doc/t5)
    start_token_id=tokenizer.pad_token_id,
    end_token_id=tokenizer.eos_token_id, 
    device=device, 
    beam_width=width, 
    temperature=temperature,
    decay_repeated=decay
  )

  # Select the best-scoring sequence out of the candidate set
  best_ids = beam_ids[0].ids()

  # Decode the token IDs back to strings
  decoded_text = tokenizer.decode(torch.tensor(best_ids), skip_special_tokens=True)

  print(f"beam search (k={width}, t={temperature}) generation results: [{decoded_text}]")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--beam', type=int, required=False, help='Specify beam search width [default: 8]')
  parser.add_argument('--decay', action='store_true', help='Apply a score decay to repeated tokens during generation')
  parser.add_argument('--input', type=str, required=True, help='Input text file')
  parser.add_argument('--greedy', action='store_true', help='Run greedy search for text generation')
  parser.add_argument('--max_length', type=int, required=False, help='Maximum token length of generated output [default: 64]')
  parser.add_argument('--prompt', type=str, required=True, help='Prompt to prefix input text with')
  parser.add_argument('--temperature', type=float, required=False, help='Temperature setting [default: 1.0]')
  args = parser.parse_args()

  device = detect_device("gpu")

  tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_ID)
  model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

  model.to(device)
  model.eval()

  text = open(args.input).read()

  if args.prompt:
    # Prepend a prompt in front of the input text
    text = f"{args.prompt}:\n" + text

  # Encode the text into a sequence of IDs
  input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

  if args.greedy:
    run_greedy_search(tokenizer, model, input_ids, device)

  if args.beam:
    run_beam_search(tokenizer, model, input_ids, device, args.beam, args.temperature or 1.0, args.decay)

if __name__ == '__main__':
  main()