import torch

class ScoredToken():
  """
  Represents a token and a corresponding score, which should roughly translate to the likelihood this
  token should be appended to some given generation sequence.
  """
  def __init__(self, token_id, score):
    self.token_id = token_id
    self.score = score

  def __str__(self):
    return f"{self.token_id}: {self.score: .8f}"
  
  def __repr__(self):
    return self.__str__()

class GeneratedSequence():
  """
  Represents a sequence in the process of being generated; an initial token, a potential end token, and a series of 
  ScoredTokens between them. This class also maintains the overall sequence score, which is the cumulative probability 
  of this generated sequence being the best output given some query.
  """
  def __init__(self, tokenizer, initial_token, end_token_id, initial_score):
    self.tokenizer = tokenizer
    self.end_token_id = end_token_id
    self._score = initial_score # Cumulative log probs of this sequence
    self.normalized_score = initial_score
    self.sequence = [ScoredToken(initial_token, initial_score)]
  
  def append(self, scored_token):
    """
    Append the given ScoredToken to this sequence; add its log-probability to this
    sequence's total cumulative log-prob
    """
    self.sequence.append(scored_token)
    self._score += scored_token.score
    self.normalized_score = self._score / len(self.sequence)

  def ids(self):
    return [st.token_id for st in self.sequence]

  def tokens(self):
    return self.tokenizer.decode(torch.tensor(self.ids()), skip_special_tokens=True)
  
  def has_ended(self):
    """
    Returns True if the last token in this sequence is the end-of-sequence token ID
    """
    return self.sequence and self.sequence[-1].token_id == self.end_token_id

  def __str__(self):
    return f"{self._score: .8f}({self.normalized_score: .8f}): {self.sequence}"

  def __repr__(self):
    return self.__str__()
  
  def __copy__(self):
    gs = GeneratedSequence(self.tokenizer, None, self.end_token_id, 0.0)
    gs.sequence = self.sequence.copy()
    gs._score = self._score
    gs.normalized_score = self.normalized_score
    return gs

  def __iter__(self):
    return self.sequence.__iter__()
  
  def __lt__(self, other_sequence):
   return self.normalized_score < other_sequence.normalized_score

  def __le__(self, other_sequence):
    return self.normalized_score <= other_sequence.normalized_score

  def __eq__(self, other_sequence):
    return self.normalized_score - other_sequence.normalized_score <= 1e-5 and self.ids() == other_sequence.ids()
  
  def __ne__(self, other_sequence):
    return self.normalized_score - other_sequence.normalized_score > 1e-5 or self.ids() != other_sequence.ids()
  
  def __gt__(self, other_sequence):
    return self.normalized_score > other_sequence.normalized_score
  
  def __ge__(self, other_sequence):
    return self.normalized_score >= other_sequence.normalized_score
