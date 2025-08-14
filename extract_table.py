import re

if __name__ == "__main__":

  raw_content = ""
  with open("reproduction/tables_3_4_5_6.txt", "r") as file:
    for line in file:
      raw_content += line[37:]
  if not raw_content:
    raise Exception("File was empty or could not be read!")

  models = re.findall(r"Model output:\n  ```\n([^`]*)```", raw_content)
  if len(models) != 5:
    raise Exception("Could not correctly extract models from log.")
  latex_code = ""
  for model in models:
    for line in model.split("\n")[:-1]:
      line = line.replace("_", "\\_")
      rate = re.findall(r"@ (.*);", line)[0]
      rate = re.sub("k", "k_{", rate)
      if "k_{" in rate:
        rate += "}"
      line = line[:line.find("@")]
      line = line.replace("->", "$\\xrightarrow{" + str(rate) + "}$")
      latex_code += line + "\n"
    latex_code += "\n"
  with open("reproduction/tables_3_4_5_6.tex", "w") as file:
    file.write(latex_code)
