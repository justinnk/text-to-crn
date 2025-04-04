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

import os

from indra.sources import reach, trips, sparser
from indra.assemblers.pysb import PysbAssembler
import pysb
from pysb.bng import generate_equations

from API import Api

from pathlib import Path

def find_file(directory, filename):
    for file in Path(directory).rglob(f"bng-linux/{filename}"):
        return str(file.parent.resolve())
    return None

BNGPATH = find_file(".venv", "BNG2.pl")
print(f"{BNGPATH=}")
os.environ["BNGPATH"] = BNGPATH

class IndraAPI(Api):
  """
  Interface with INDRA.
  """

  def __init__(
    self,
    device: str = "cuda",
  ):
    super().__init__(
      "",
      device,
      None,
      None,
      "data/V11.0-1000/train_wo_meta.json",
      False,
      False,
      0,
      False
    )

  def destroy(self):
    pass
  
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
    if len(messages) > 0:
      raise NotImplementedError("INDRA API does not support interaction.")
    messages.append({"role": "user", "content": prompt})
    tp = trips.process_text(prompt)
    pa = PysbAssembler()
    pa.add_statements(tp.statements)
    model = pa.make_model(policies='two_step')
    model_str = ""
    if len(model.rules) > 0:
      generate_equations(model)
      def species_to_str(id):
        species = model.species[id]
        species_name = ""
        for monomer_pattern in species.monomer_patterns:
          species_name += monomer_pattern.monomer.name
        return species_name
      model_str = "```\n"
      for idx, reaction in enumerate(model.reactions):
        print(reaction)
        model_str += "+".join(map(species_to_str, reaction["reactants"]))
        model_str += " -> "
        model_str += "+".join(map(species_to_str, reaction["products"]))
        model_str += " @ "
        if isinstance(reaction["rate"], pysb.core.Parameter):
          model_str += str(reaction["rate"].value)
        else:
          model_str += f"k{idx}"
        model_str += ";\n"
      model_str += "```"
    messages.append({
      "role": "assistant",
      "content": model_str
    })
    return model_str, messages

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
