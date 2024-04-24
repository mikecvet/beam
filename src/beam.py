from copy import deepcopy
import math
from sequence import GeneratedSequence, ScoredToken
import torch
import tqdm
from collections import Counter
import torch.nn as nn

def search(tokenizer, model, src, start_token_id, end_token_id, device, beam_width=8, temperature=1.0, max_length=64):
  model.eval()

  print(f"beginning beam search | k = {beam_width} bos = {start_token_id} eos = {end_token_id} temp = {temperature} beam_width = {beam_width}")

  candidate_sequences = [GeneratedSequence(tokenizer, start_token_id, end_token_id, 0.0)]

  for i in tqdm.tqdm(range(max_length)):
    # Temporary list to store candidates for the next generation step
    next_step_candidates = []

    for candidate in candidate_sequences:
      if not candidate.has_ended(): # skip candidate sequences which have included the end-of-sequence token
        tgt = torch.tensor(candidate.ids(), device=device).unsqueeze(0)
        output = model(src, decoder_input_ids=tgt)
        logits = output.logits[:, -1, :]

        scaled_logits = logits / temperature
        
        probs = torch.softmax(scaled_logits, dim=-1)

        #print(f"ids: {ids} logits: {logits} \nscaled_logits {scaled_logits} \nprobs: {probs}")

        top_probs, top_ids = probs.topk(beam_width)

        print(f"{candidate.score}: [{candidate.tokens()}], next token probabilities:")
        for p, w in zip(top_probs.tolist()[0], tokenizer.convert_ids_to_tokens(top_ids.tolist()[0])):
          print(f"\tp: {p: .8f}: {w}")
        print("\n")

        token_counts = Counter(t.token_id for t in candidate)

        # Generate new candidates and add them to candidates list
        for i in range(beam_width):
          next_token_id = top_ids[:, i].item()
          next_score = torch.log(top_probs[:, i]).item()

          if next_token_id in token_counts:
            count = token_counts[next_token_id]
            decay = 1 + math.log(count + 1)
            #print(f"{tokenizer.convert_ids_to_tokens([next_token_id])} count: {count} decay: {decay}, score: {next_score}, next: {next_score * decay}")
            #next_score *= decay

          new_seq = deepcopy(candidate)
          new_seq.append(ScoredToken(next_token_id, next_score))

          next_step_candidates.append(new_seq)
      else:
        next_step_candidates.append(candidate)

    print(f"next step candidates:")
    for seq in reversed(sorted(next_step_candidates)):
      print(f"\t{seq.score: .8f}: [{seq.tokens()}]")
    print("\n")

    next_step_candidates.sort()
    candidate_sequences = list(reversed(next_step_candidates))[:beam_width]

    # Break if all sequences in the heap end with the eos_token_id
    if all(seq.has_ended() for seq in candidate_sequences):
      break

  for candidate in candidate_sequences:
    # Normalize scores by length, to avoid penalizing longer sequences
    candidate.normalize_score()

  return candidate_sequences
