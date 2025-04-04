"""
Copyright 2025 Justin Kreikemeyer, Miłosz Jankowski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
 
  The above copyright notice and this permission notice shall be included in all copies or
  substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

from abc import ABC, abstractmethod
import json

SYSTEM_PROMPT = (
  "You are a translator that translates from natural language descriptions to formal reaction system simulation models."
  "Do not generate anything except the formal output. Do not provide any explanation. Closely adhere to the provided example syntax."
  "When mentioning entities in the formal model, try to match their names precisely to their mentions in the textual description."
)


class Api(ABC):
  """
  Class for interacting with LLMs.
  """

  LORA_DIR = "loras"

  def __init__(
    self,
    model_name: str = "",
    device: str = "cuda",
    grammar_path: str = None,
    lora_path: str = None,
    few_shot_dataset_path: str = "data/V9.1-1000/train_wo_meta.json",
    use_few_shot: bool = True,
    use_sys_promt: bool = True,
    num_few_shot_examples: int = 10,
    embed_examples: bool = False
  ):
    self.model_name = model_name
    self._device = device
    # system message / few shot
    self.use_few_shot = use_few_shot
    self.use_sys_prompt = use_sys_promt
    self.few_shot_dataset_path = few_shot_dataset_path
    self.num_few_shot_examples = num_few_shot_examples
    self.embed_examples = embed_examples
    self._few_shot_prompt = []
    #if self.num_few_shot_examples > 0:
    if num_few_shot_examples == 0 and use_few_shot:
      print("Warning: Using few shot with 0 examples will just include system prompt!")
    if num_few_shot_examples > 0 and not use_few_shot:
      print("Warning: Set few shot example number but use_few_shot is False!")
    self._build_few_shot_prompt()
    print(self._few_shot_prompt)

  def _build_few_shot_prompt(self):
    few_shot_data = {}
    with open(self.few_shot_dataset_path, "r") as file:
      few_shot_data = json.load(file)
    if self.use_sys_prompt:
      self._few_shot_prompt = [{
        "role": "system",
        "content": SYSTEM_PROMPT
      }]
    if not self.use_few_shot: return
    if self.num_few_shot_examples <= 0: return
    # whether examples are embedded in the first prompt or included as conversation
    if self.embed_examples:
      examples = "Here are some example interactions:\n"
      for idx in range(self.num_few_shot_examples):
        for cidx in range(len(few_shot_data[idx])):
          examples += "User:\n" + few_shot_data[idx][cidx]["instruction"]
          examples += "\nAssistant:\n" + few_shot_data[idx][cidx]["output"]
          examples += "\n"
      self._few_shot_prompt.extend([
        {"role": "user", "content": examples},
        {"role": "assistant", "content": "Ok."}
      ])
    else:
      for idx in range(self.num_few_shot_examples):
        for cidx in range(len(few_shot_data[idx])):
          self._few_shot_prompt.extend([
            {"role": "user", "content": few_shot_data[idx][cidx]["instruction"]},
            {"role": "assistant", "content": few_shot_data[idx][cidx]["output"]}
          ])

  @abstractmethod
  def destroy(self):
    raise NotImplementedError()
  
  @abstractmethod
  def apply_lora(self, lora_path: str):
    raise NotImplementedError()
  
  @abstractmethod
  def chat(
    self,
    messages,
    prompt,
    seed: int,
    temperature: float = 1.0,
    max_new_tokens: int = 300
  ):
    raise NotImplementedError()

  def get_reaction_system_for_prompt(self, prompt: str, seed: int, temperature: float = 1.0, max_new_tokens: int = 300):
    """Use (pre-prompted) LLM to generate a reaction system for the given prompt."""
    return self.chat([], prompt, seed, temperature, max_new_tokens)[0]
  
  @abstractmethod
  def train_lora(
    self,
    name: str,
    dataset: str,
    *,
    alpha: int = 384,
    rank: int = 384,
    gradient_accumulation=80,
    batch_size=10,
    lora_dropout: float = 0.1,
    lr: float = 1.0e-4,
    seed: int = 42,
    train_with_sys_prompt: bool = False,
    resume_from_checkpoint: bool = False
  ):
    raise NotImplementedError()
  
