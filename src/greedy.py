import torch

def search(model, input_ids, max_length, start_token_id, end_token_id, device):

  # The output sequence starts as a tensor containing just the start token ID
  output_sequence = torch.full((1, 1), start_token_id, device=device)

  for _ in range(max_length - 1):
    # Predict next token, given input context IDs and output sequence so far
    output = model(input_ids=input_ids, decoder_input_ids=output_sequence)

    # Extract the raw model output
    logits = output.logits[:, -1, :]

    # Extract the argmax, the highest-scoring element, from the model output logits.
    # Unsqueeze to remove the singular batch dimension.
    next_token_id = logits.argmax(-1).unsqueeze(-1)

    # Concatenate the next token ID into the output sequence
    output_sequence = torch.cat([output_sequence, next_token_id], dim=-1)

    # If the next token happens to be the end-of-sequence token, break the loop
    if (next_token_id == end_token_id):
      break

  # Return greedily-computed output token sequence
  return output_sequence