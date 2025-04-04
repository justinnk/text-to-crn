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


"""
Functions to generate reactions of certain types according to the standards in different domains.
In order to be discovered, they must follow the naming scheme _gen_{domain}_{type}(...)
"""

import random
from copy import deepcopy

class DomainSpecificReactionGenerator:

  def __init__(self, rng: random.Random):
    self.rng = rng

  def get_reactions(self, domain: str, construct: str, species_names: list[str]):

    ### Systems Biology ###
    def _gen_bio_general():
      return self._return_type(construct, [self._general_reaction(species_names)])
    def _gen_bio_degradation():
      return self._return_type(construct, [self._degradation_reaction(species_names)])
    def _gen_bio_degradation_multi():
      return self._return_type(construct, self._multi_reactions(species_names, self._degradation_reaction))
    def _gen_bio_degradation_relational():
      reaction = self._degradation_reaction(species_names)
      reaction["rate"] = self._get_rate(force_num=True)
      return self._return_type(construct, [reaction])
    def _gen_bio_production():
      return self._return_type(construct, [self._production_reaction(species_names)])
    def _gen_bio_production_multi():
      return self._return_type(construct, self._multi_reactions(species_names, self._production_reaction))
    def _gen_bio_production_relational():
      reaction = self._production_reaction(species_names)
      reaction["rate"] = self._get_rate(force_num=True)
      return self._return_type(construct, [reaction])
    def _gen_bio_complexation():
      reactant1 = self.rng.choice(species_names)
      reactant2 = self.rng.choice(species_names)
      product = reactant1 + reactant2 # complex
      return self._return_type(construct, [dict(
        reactants=[reactant1, reactant2],
        products=[product],
        rate=self._get_rate()
      )])
    def _gen_bio_chain():
      if len(species_names) < 3:
        raise ValueError("Cannot construct chain for less than three species.")
      num_intermediates = self.rng.randint(1, len(species_names) - 2)
      valid_species_names = deepcopy(species_names)
      intermediates = []
      start = ""
      end = ""
      reactions = []
      for idx in range(num_intermediates + 2):
        if idx == 0:
          start = self.rng.choice(valid_species_names)
          valid_species_names.remove(start)
        elif idx == num_intermediates + 2 - 1:
          end = self.rng.choice(valid_species_names)
          valid_species_names.remove(end)
        else:
          intermediates.append(self.rng.choice(valid_species_names))
          valid_species_names.remove(intermediates[-1])
      for idx in range(num_intermediates + 1):
        if idx == 0:
          reactions.append(dict(
            reactants=[start],
            products=[intermediates[0]],
            rate="k"
          ))
        elif idx == num_intermediates:
          reactions.append(dict(
            reactants=[intermediates[-1]],
            products=[end],
            rate="k"
          ))
        else:
          reactions.append(dict(
            reactants=[intermediates[idx-1]],
            products=[intermediates[idx]],
            rate="k"
          ))
      return self._return_type(construct, reactions, intermediates=intermediates)
        
    def _gen_bio_catalysis():
      valid_species_names = deepcopy(species_names)
      catalyst = self.rng.choice(valid_species_names)
      valid_species_names.remove(catalyst)
      reactant1 = self.rng.choice(valid_species_names)
      valid_species_names.remove(reactant1)
      product1 = self.rng.choice(valid_species_names)
      return self._return_type(
        construct,
        [dict(reactants=[reactant1, catalyst], products=[product1, catalyst], rate=self._get_rate())],
        reactants1=reactant1,
        reactants2=catalyst,
        products1=product1,
        products2=catalyst
      )
    def _gen_bio_conversion():
      if len(species_names) < 4:
        raise ValueError("For conversion, at least four species are needed.")
      chosen_species_names = self.rng.sample(species_names, k=4)
      return self._return_type(construct, [self._general_reaction(chosen_species_names, min_reactants=2, min_products=2, max_reactants=2, max_products=2)])

    ### Ecology ###
    def _gen_eco_general():
      return self._return_type(construct, [self._general_reaction(species_names)])
    def _gen_eco_degradation():
      return self._return_type(construct, [self._degradation_reaction(species_names)])
    def _gen_eco_degradation_multi():
      return self._return_type(construct, self._multi_reactions(species_names, self._degradation_reaction))
    def _gen_eco_degradation_relational():
      reaction = self._degradation_reaction(species_names)
      reaction["rate"] = self._get_rate(force_num=True)
      return self._return_type(construct, [reaction])
    def _gen_eco_production():
      return self._return_type(construct, [self._production_reaction(species_names)])
    def _gen_eco_production_multi():
      return self._return_type(construct, self._multi_reactions(species_names, self._production_reaction))
    def _gen_eco_production_relational():
      reaction = self._production_reaction(species_names)
      reaction["rate"] = self._get_rate(force_num=True)
      return self._return_type(construct, [reaction])
    def _gen_eco_mating():
      partner1 = self.rng.choice(species_names)
      partner2 = partner1 + "_male"
      offspring = partner1 + "_pup"
      partner1 = partner1 + "_female"
      reactants = [partner1, partner2]
      products = [partner1, partner2, offspring]
      self.rng.shuffle(reactants)
      self.rng.shuffle(products)
      return self._return_type(
         construct,
         [dict(reactants=reactants, products=products, rate=self._get_rate())],
         offspring=offspring
      )
    def _gen_eco_predation():
      valid_species_names = deepcopy(species_names)
      predator = self.rng.choice(valid_species_names)
      valid_species_names.remove(predator)
      prey = self.rng.choice(valid_species_names)
      reactants = [predator, prey]
      products = [predator]
      self.rng.shuffle(reactants)
      return self._return_type(
        construct,
        [dict(reactants=reactants, products=products, rate=self._get_rate())],
        reactants1=predator,
        reactants2=prey
      )
    def _gen_eco_chain():
      return _gen_bio_chain()

    ### Epidemiology ###
    def _gen_epi_general():
      return self._return_type(construct, [self._general_reaction(species_names, min_reactants=1, max_reactants=1, min_products=1, max_products=1)])
    def _gen_epi_degradation():
      return self._return_type(construct, [self._degradation_reaction(species_names)])
    def _gen_epi_degradation_multi():
      return self._return_type(construct, self._multi_reactions(species_names, self._degradation_reaction))
    def _gen_epi_degradation_relational():
      reaction = self._degradation_reaction(species_names)
      reaction["rate"] = self._get_rate(force_num=True)
      return self._return_type(construct, [reaction])
    def _gen_epi_production():
      return self._return_type(construct, [self._production_reaction(species_names)])
    def _gen_epi_production_multi():
      return self._return_type(construct, self._multi_reactions(species_names, self._production_reaction))
    def _gen_epi_production_relational():
      reaction = self._production_reaction(species_names)
      reaction["rate"] = self._get_rate(force_num=True)
      return self._return_type(construct, [reaction])

    # find the function to use in the locals()
    fun_name = f"_gen_{domain}_{construct}"
    if fun_name in locals():
      return locals()[fun_name]()
    else:
      print(domain, construct, fun_name, locals())
      raise ValueError(f"There is no implementation for the construct {construct} in the {domain} domain.")

  def _get_rate(self, force_var: bool = False, force_num: bool = False) -> float|str:
    if force_var and force_num: raise ValueError("Cannot force var *and* num.")
    if force_var: return "k"
    if force_num: return str(round(10 * self.rng.random() + 0.01, 2))
    include_rate = bool(self.rng.randint(0, 1))
    return str(round(10 * self.rng.random() + 0.01, 2)) if include_rate else "k" 

  def _general_reaction(
        self,
        species_names: set,
        min_reactants: int = 1,
        min_products: int = 1,
        max_reactants: int = 1,
        max_products: int = 3
  ) -> dict:
    num_reactants = self.rng.randint(min_reactants, max_reactants)
    num_products = self.rng.randint(min_products, max_products)
    # sample reactants and products and make sure they are not equal
    reactants=[self.rng.choice(species_names) for _ in range(num_reactants)]
    products=[self.rng.choice(species_names) for _ in range(num_products)]
    while reactants == products:
      reactants=[self.rng.choice(species_names) for _ in range(num_reactants)]
      products=[self.rng.choice(species_names) for _ in range(num_products)]
    return dict(
      reactants=reactants,
      products=products,
      rate=self._get_rate()
    )
  
  def _degradation_reaction(self, species_names: set):
    return self._general_reaction(
      species_names,
      min_reactants=1,
      max_reactants=1,
      min_products=0,
      max_products=0
    )
  
  def _production_reaction(self, species_names: set):
    return self._general_reaction(
      species_names,
      min_reactants=0,
      max_reactants=0,
      min_products=1,
      max_products=1
    )
  
  def _multi_reactions(self, species_names, reac_type_func: callable):
    if reac_type_func not in [self._degradation_reaction, self._production_reaction]:
      raise NotImplementedError("_multi_reactions only works for degradation and production at the moment.")
    if len(species_names) < 2:
       raise ValueError("Cannot construct multiple reactions with less than two species.")
    num_reactions = self.rng.randint(2, min(3, len(species_names)))
    valid_species_names = deepcopy(species_names)
    reactions = []
    for _ in range(num_reactions):
      reaction = reac_type_func(valid_species_names)
      reaction["rate"] = self._get_rate(force_var=True)
      taken_species = reaction["reactants"] + reaction["products"]
      reactions.append(reaction)
      for s in taken_species:
        valid_species_names.remove(s)
    return reactions

  @staticmethod
  def _return_type(
    construct: str,
    reactions: list[dict],
    intermediates: list[str] = [],
    offspring: str = "",
    reactants1: str = "",
    reactants2: str = "",
    products1: str = "",
    products2: str = ""
  ) -> dict:
    return dict(
     construct=construct,
     reactions=reactions,
     intermediates=intermediates,
     offspring=offspring,
     reactants1=reactants1,
     reactants2=reactants2,
     products1=products1,
     products2=products2,
     species_names=list(set([name for names in [reac["reactants"] + reac["products"] for reac in reactions] for name in names]))
    )
  