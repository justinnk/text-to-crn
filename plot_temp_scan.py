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


import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "ieee"])

ALL_COLORS = ["#00843D","#78BE20","#772583","#981D97","#FF6600","#FFA300","#00566e","#009CA6"]

one_shot_wo_grammar = pd.read_csv("results/temp_scan/one-shot/summary-wo_grammar.csv")
one_shot_w_grammar = pd.read_csv("results/temp_scan/one-shot/summary-w_grammar.csv")
few_shot_wo_grammar = pd.read_csv("results/temp_scan/few-shot/summary-wo_grammar.csv")
few_shot_w_grammar = pd.read_csv("results/temp_scan/few-shot/summary-w_grammar.csv")
lora_wo_grammar_zero_shot = pd.read_csv("results/temp_scan/lora-v11.0-1000-zero-shot/summary-wo_grammar.csv")
lora_w_grammar_zero_shot = pd.read_csv("results/temp_scan/lora-v11.0-1000-zero-shot/summary-w_grammar.csv")
lora_wo_grammar_few_shot = pd.read_csv("results/temp_scan/lora-v11.0-1000-few-shot/summary-wo_grammar.csv")
lora_w_grammar_few_shot = pd.read_csv("results/temp_scan/lora-v11.0-1000-few-shot/summary-w_grammar.csv")

fig, ax = plt.subplots(figsize=(3, 3))

ax.errorbar(one_shot_wo_grammar.temperature, one_shot_wo_grammar["mean"], yerr=one_shot_wo_grammar["stddev"], label="One Shot", color=ALL_COLORS[0], ls="-")
ax.errorbar(one_shot_w_grammar.temperature, one_shot_w_grammar["mean"], yerr=one_shot_w_grammar["stddev"], label="One Shot (GCD)", color=ALL_COLORS[1], ls="-")
ax.errorbar(few_shot_wo_grammar.temperature, few_shot_wo_grammar["mean"], yerr=few_shot_wo_grammar["stddev"], label="Few Shot", color=ALL_COLORS[2], ls="-")
ax.errorbar(few_shot_w_grammar.temperature, few_shot_w_grammar["mean"], yerr=few_shot_w_grammar["stddev"], label="Few Shot (GCD)", color=ALL_COLORS[3], ls="-")
ax.errorbar(lora_wo_grammar_zero_shot.temperature, lora_wo_grammar_zero_shot["mean"], yerr=lora_wo_grammar_zero_shot["stddev"], label="Zero Shot + LoRA", color=ALL_COLORS[4], ls="-")
ax.errorbar(lora_w_grammar_zero_shot.temperature, lora_w_grammar_zero_shot["mean"], yerr=lora_w_grammar_zero_shot["stddev"], label="Zero Shot + LoRA (GCD)", color=ALL_COLORS[5], ls="-")
ax.errorbar(lora_wo_grammar_few_shot.temperature, lora_wo_grammar_few_shot["mean"], yerr=lora_wo_grammar_few_shot["stddev"], label="Few Shot + LoRA", color=ALL_COLORS[6], ls="-")
ax.errorbar(lora_w_grammar_few_shot.temperature, lora_w_grammar_few_shot["mean"], yerr=lora_w_grammar_few_shot["stddev"], label="Few Shot + LoRA (GCD)", color=ALL_COLORS[7], ls="-")
#ax.hlines(xmin=0, xmax=1.0, y=1.0, color="black")
ax.set_ylim((-0.05, 1.05))
ax.set_xticks([],[],minor=True)
ax.set_xlabel("Decoding Temperature")
ax.set_ylabel("Accuracy")
#ax.set_title("Accuracy at Different Temperatures")
ax.legend(frameon=True, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig("exp_temp_scan.pdf", dpi=300)


plt.cla()
plt.clf()


lora_wo_grammar_few_shot_embedded = pd.read_csv("results/temp_scan/lora-v11.0-1000-few-shot-embedded/summary-wo_grammar.csv")
lora_w_grammar_few_shot_embedded = pd.read_csv("results/temp_scan/lora-v11.0-1000-few-shot-embedded/summary-w_grammar.csv")
#lora_wo_grammar_few_shot_extra = pd.read_csv("results/temp_scan/lora-v11.0-1000-few-shot-extra/summary-wo_grammar.csv")
#lora_w_grammar_few_shot_extra = pd.read_csv("results/temp_scan/lora-v11.0-1000-few-shot-extra/summary-w_grammar.csv")

fig, ax = plt.subplots(figsize=(3, 3))

ax.errorbar(lora_wo_grammar_few_shot_embedded.temperature, lora_wo_grammar_few_shot_embedded["mean"], yerr=lora_wo_grammar_few_shot_embedded["stddev"], label="Few Shot + LoRA (emb.)", color=ALL_COLORS[4], ls="-")
ax.errorbar(lora_w_grammar_few_shot_embedded.temperature, lora_w_grammar_few_shot_embedded["mean"], yerr=lora_w_grammar_few_shot_embedded["stddev"], label="Few Shot + LoRA (GCD, emb.)", color=ALL_COLORS[5], ls="-")
# original curves
ax.errorbar(lora_wo_grammar_zero_shot.temperature, lora_wo_grammar_zero_shot["mean"], yerr=lora_wo_grammar_zero_shot["stddev"], label="Zero Shot + LoRA", color=ALL_COLORS[6], ls="--")
#ax.errorbar(lora_w_grammar_zero_shot.temperature, lora_w_grammar_zero_shot["mean"], yerr=lora_w_grammar_zero_shot["stddev"], label="Zero Shot + LoRA (GCD)", color=ALL_COLORS[5], ls="--")
ax.errorbar(lora_wo_grammar_few_shot.temperature, lora_wo_grammar_few_shot["mean"], yerr=lora_wo_grammar_few_shot["stddev"], label="Few Shot + LoRA", color=ALL_COLORS[4], ls="--")
#ax.errorbar(lora_w_grammar_few_shot.temperature, lora_w_grammar_few_shot["mean"], yerr=lora_w_grammar_few_shot["stddev"], label="Few Shot + LoRA (GCD)", color=ALL_COLORS[7], ls="--")
ax.set_ylim((-0.05, 1.05))
ax.set_xticks([],[],minor=True)
ax.set_xlabel("Decoding Temperature")
ax.set_ylabel("Accuracy")
#ax.set_title("Accuracy at Different Temperatures (further experiments)")
#ax.legend(frameon=True, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.legend(frameon=True, loc="lower center")
fig.savefig("exp_temp_scan_extra.pdf", dpi=300)