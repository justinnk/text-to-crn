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


from API import Api

import gc

from openai import OpenAI


class ChatGptAPI(Api):
  """
  Interface with an OpenAI model using the OpenAI API.
  """

  def __init__(
    self,
    model_name: str = "gpt-4o-2024-08-06",
    device: str = "cuda",
    grammar_path: str = None,
    lora_path: str = None,
    few_shot_dataset_path: str = "data/V11.0-1000/train_wo_meta.json",
    use_few_shot: bool = True,
    use_sys_prompt: bool = True,
    num_few_shot_examples: int = 10,
    embed_examples: bool = False
  ):
    super().__init__(
      model_name,
      device,
      grammar_path,
      lora_path,
      few_shot_dataset_path,
      use_few_shot,
      use_sys_prompt,
      num_few_shot_examples,
      embed_examples
    )
    self.client = OpenAI(
      organization="INSERT HERE",
      project="INSERT HERE",
    )
    if grammar_path is not None or lora_path is not None:
      raise NotImplementedError()

  def destroy(self):
    self.client.close()
    del self.client
    gc.collect()
  
  def apply_lora(self, lora_path: str):
    raise NotImplementedError()
  
  def chat(
    self,
    messages,
    prompt,
    seed: int,
    temperature: float = 1.0,
    max_new_tokens: int = 300
  ):
    messages.append({"role": "user", "content": prompt})
    _messages = self._few_shot_prompt + messages
    response = self.client.chat.completions.create(
      model=self.model_name,
      messages=_messages,
      temperature=temperature,
      max_completion_tokens=max_new_tokens,
      seed=seed
    )
    messages.append({
      "role": "assistant",
      "content": response.choices[0].message.content
    })
    return response.choices[0].message.content, messages

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


if __name__ == "__main__":
  api = ChatGptAPI()
  prompt = "The interaction of susceptible with infected individuals results in two infected individuals at a rate of 0.02. Infected individuals recover at a rate of 5."
  out = api.get_reaction_system_for_prompt(prompt, seed=42)
  print(out)