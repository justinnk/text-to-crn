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

KINMOD_SYS_PROMPT="""You are a program that converts biochemical reactions written in natural language into Antimony language. First, remember the following conversion rules.

# Conversion rules
| Natural language | Antimony language | 
| E catalyzes the conversion of X to Y | X -> Y ; kcat_E_X_Y * E * X / ( Km_E_X_Y + X ) ; kcat_E_X_Y = 1 ; Km_E_X_Y = 1; E = 1 |
| X is phosphorylated | X -> X_P ; Vp_X * X / ( Km_X + X ) ; Vp_X = 1 ; Km_X = 1 |
| X is converted into Y | X -> Y ; kc_X_Y * X; kc_X_Y = 1 |
| X and Y bind to form Z | X + Y -> Z ; ka_X_Y_Z * X * Y ; ka_X_Y_Z = 1 |
| X dissociates into Y and Z | X -> Y + Z ; kd_X_Y_Z * X ; kd_X_Y_Z = 1 |
| X is produced (or transcribed) | -> X ; km_X ; km_X  = 1 |
| Expression of X is repressed (or negatively regulated or downregulated) by R | -> X ; km_X_R * K_X_R ^ n_X_R / ( K_X_R ^ n_X_R + R ^ n_X_R ) ; km_X_R = 1 ; K_X_R = 1 ; n_X_R = 1 |
| Expression of X is activated (or positively regulated or upregulated) by A | -> X ; km_X_A * A ^ n_X_A / ( K_X_A ^ n_X_A + A ^ n_X_A ) ; km_X_A = 1 ; K_X_A = 1 ; n_X_A = 1 |
| Y is translated from X | -> Y ; kp_X_Y * X ; kp_X_Y = 1 |
| X degrades (or decays) | X -> ; kdeg_X * X ; kdeg_X = 1 |
| X (concentration) is Y M (or mM or uM or nM or pM) | X = Y |

# Examples
"The expression of G is negatively regulated by R." is converted into "-> G ; km_G_R * K_G_R ^ n_G_R / ( K_G_R ^ n_G_R + R ^ n_G_R ) ; km_G_R = 1 ; K_G_R = 1 ; n_G_R = 1"
"G is upregulated by A." is converted into "-> G ; km_G_A * A ^ n_G_A / ( K_G_A ^ n_G_A + A ^ n_G_A ) ; km_G_A = 1 ; K_G_A = 1 ; n_G_A = 1"

Using the conversion rules provided, convert the biochemical reactions listed below into Antimony language. After converting each reaction, create a bullet point list that includes all the resulting expressions. In the list, show one reaction per line. No need to provide further explanations, just present the list. Start each line with '-'.
"""

OUR_SYS_PROMPT="""You are a program that converts biochemical reactions written in natural language into a formal reaction language. First, remember the following conversion rules.

# Conversion rules
| Natural language | Antimony language | 
| E catalyzes the conversion of X to Y | X + E -> Y + E @ k0; |
| X is converted into Y | X -> Y @ k0; |
| X and Y bind to form Z | X + Y -> Z @ k0; |
| X dissociates into Y and Z | X -> Y + Z @ k0; |
| X is produced (or transcribed) | -> X @ k0; |
| X degrades (or decays) | X -> @ k0; |

# Examples
"The following describes a reaction system. Please translate to a formal description. RPL35A is produced. It is produced with a rate of 2.42. In addition, GPM1 and RPL35A are removed from the system. RPL35A emerges at a rate of 7.9." is converted to ```\n -> RPL35A @ 2.42;\nGPM1 ->  @ k0;\nRPL35A ->  @ k1;\n -> RPL35A @ 7.9;\n```
"The following describes a reaction system. Please translate to a formal description. HSP26 vanishes. It leaves the system at a rate of 9.82. Two ATP are the result of a conversion of TDH3 and TPI1. A chain reaction occurs from TDH3 through HSP26, TPI1, and ATP to GPM1. The complex ATPGPM1 forms from ATP and GPM1." is converted to ```\nHSP26 ->  @ 9.82;\nTDH3 + TPI1 -> 2ATP @ k0;\nTDH3 -> HSP26 @ k1;\nHSP26 -> TPI1 @ k2;\nTPI1 -> ATP @ k3;\nATP -> GPM1 @ k4;\nATP + GPM1 -> ATPGPM1 @ k5;\n```

Using the conversion rules provided, convert the biochemical reactions listed below into the formal language. After converting each reaction, put them into a code block marked with ```. Inside the code block, show one reaction per line. No need to provide further explanations, just present the model.
"""

class KinModGptAPI(Api):
  """
  Interface with an OpenAI model using the OpenAI API.
  """

  def __init__(
    self,
    model_name: str = "gpt-4o-2024-08-06",
    device: str = "cuda",
    sys_prompt: str = KINMOD_SYS_PROMPT
  ):
    super().__init__(
      model_name,
      device,
      None,
      None,
      "data/V11.0-1000/train_wo_meta.json",
      False,
      False,
      0,
      False
    )
    self.sys_prompt = sys_prompt
    self.client = OpenAI(
      organization="org-stTiZ39vON2q7x7P9XpjXX7f",
      project="proj_pYnjF76De1a0m3frYun1nybm",
    )

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
    _messages = [{"role": "system", "content": self.sys_prompt}] + messages
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
