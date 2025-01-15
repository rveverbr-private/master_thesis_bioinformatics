#!/usr/bin/env python
# coding: utf-8


from IPython import get_ipython


if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import scanpy as sc
import latenta as la
import jax
import jax.numpy as jnp
import optax
import tqdm.auto as tqdm
import scipy
import random
import re
import dill as pickle
import sklearn.decomposition
import os
import eyck


# Subclass random.Random to add a custom key() method
class KeyedRandom(random.Random):
    def __init__(self, x=None, bit_amount=16):
        super().__init__(x)
        self.bit_amount = bit_amount

    def key(self, bit_amount=None):
        # Use the provided bit_amount, or default to self.bit_amount if None
        bit_amount = bit_amount if bit_amount is not None else self.bit_amount
        return self.getrandbits(bit_amount)


# Create an instance of the custom generator with a fixed seed
generator = KeyedRandom(42)


def create_base_model(adata, definition_type=int):
    # create description based on adata object
    transcriptome_definition = la.Definition([la.Dim(adata.obs.index.astype(definition_type), name = "cell"), la.Dim(adata.var.index.astype(definition_type), name = "gene")])

    # create a mu with value 0 for now
    mu = la.modular.Multiplicative(definition = transcriptome_definition.clean, label = "mu")

    # create a distribution that we believe our mRNA counts will follow.
    transcriptome_distribution = la.distributions.NegativeBinomial2(mu = mu, definition = transcriptome_definition, key=generator.key())

    # create a dispersion parameter for the transcriptome distribution
    dispersion = la.Latent(
        p=la.distributions.Normal(
            la.Parameter(0.0), la.Fixed(1.0, transforms=[la.transforms.Exp()])
        ),
        q=la.distributions.Normal(
            loc= la.Parameter(0.1, definition = transcriptome_definition.clean[1]),
            scale= la.Parameter(0.1,  transforms=[la.transforms.Exp()]),
        transforms=[la.transforms.Exp()]),
        key=generator.key()
    )
    transcriptome_distribution.dispersion = dispersion
        # use a loader. This allows the large dataset to only load in parts of the data on demand, preventing large RAM usage.
    try:
        loader = la.variables.loaders.MemoryLoader(
            value = adata.layers["counts"],
            definition = transcriptome_definition
        )
    except ValueError:
        adata.layers["counts"] = adata.layers["counts"].toarray()
        loader = la.variables.loaders.MemoryLoader(
            value = adata.layers["counts"],
            definition = transcriptome_definition
        )
    loader.initialize(transcriptome_definition)

    # actual data as latenta variable.
    transcriptome_observation = la.Observation(value = loader, p = transcriptome_distribution, label = "transcriptome", definition = transcriptome_definition)

    # create a variable for the library size per cell. This is to calculate the the mu for the distribution
    lib = la.Fixed(np.array(adata.layers["counts"].sum(1)), definition = transcriptome_definition[0], label = "lib")

    # calculate the baseline values
    baseline_value = np.array(((adata.layers["counts"].sum(0)+1) / (adata.layers["counts"].sum(0)+1).sum()))
    baseline = la.Parameter(
        np.log(baseline_value),
        definition=transcriptome_definition.clean[1],
        label = "baseline",
    )
    transcriptome_distribution.dispersion = dispersion
    fc = la.modular.Additive(
        definition = transcriptome_definition,
        label = "lfc",
        subsettable = [transcriptome_definition[1]]
    )
    rho = la.modular.Additive(definition=transcriptome_definition.clean, label="rho", transforms = [la.transforms.Exp()])
    rho.lfc = fc
    rho.baseline = baseline
    mu.rho = rho
    mu.lib = lib
    transcriptome_observation.rootify()
    return transcriptome_observation


def create_model(model, optimizer, architecture, model_name):
    model = la.models.reinitialize_keys(model, generator.key())

    # create minibatcher
    k = la.utils.key.StatefulKey(generator.key())
    minibatcher = la.train.minibatching.Minibatcher(model.clean[0], k(), size=1000)
    minibatcher_validation = la.train.minibatching.Minibatcher(model.clean[0], jax.random.key(1), size=1000)
    program = la.train.svi.SVI(
        model,
        optimizer,
        minibatcher=minibatcher,
        minibatcher_validation = minibatcher_validation,
        
    )
    model_info = {
        "architecture": architecture,
        "model_name":model_name
    }
    return model, program, model_info



def train_model(program, iterations:int=1000):
    losses = []
    losses_model = program.train(iterations, progress=True)
    losses.append(losses_model)
    losses = pd.DataFrame(np.array(losses))
    return losses



def create_program(model, program=None, minibatcher=None):
    if not program:
        program = la.Program()
    leaves = model.findall(lambda x: la.variables.leaf.is_leaf(x))
    for leaf in leaves:
        leaf.loader.load(program, minibatcher=minibatcher)
    observations = model.findall(lambda x: la.utils.isinstance2(x, la.Observation))
    latents = model.findall(lambda x: la.utils.isinstance2(x, la.Latent))
    nns = model.findall(lambda x: la.utils.isinstance2(x, la.links.nn.NN))
    for v in [*observations, *latents]:
        v.elbo(program)
    for v in [*observations]:
        v.lik(program)
    for v in [*nns]:
        v.run(program, train=True)
    return program



def comparison(model, program, runner,model_info):
    liks = []
    elbos = []
    elbos_std = []

    k = la.utils.key.StatefulKey(generator.key())

    # set all keys
    key_loaders = []
    keys = model.findall(lambda x: la.utils.isinstance2(x, la.variables.key.Key))
    for key in keys:
        key.loader.value = k()
    reloader = program.reloader()
    state = program.create_state(device=jax.devices()[0])
    # get elbos
    elbos_model = []
    liks_model = []
    output = runner(state)
    elbos_model.append(output[("elbo",)])
    liks_model.append(output[("lik",)])
    state = reloader(state)
    elbos_model = np.array(elbos_model)
    liks_model = np.array(liks_model)
    elbos.append(elbos_model.mean())
    elbos_std.append(elbos_model.std())
    liks.append(liks_model.mean())
    scores = pd.DataFrame(
        {
            "elbo": np.array(elbos),
            "elbo_std": np.array(elbos_std),
            "lik": np.array(liks),
            "penalty": np.array(liks) - np.array(elbos),
            "architecture": model_info["architecture"],
            "model_name": model_info["model_name"]
        }
    )
    return scores



def latenta_processes_basic(model, optimizer, architecture, model_name, training_iterations:int=1000):
    model_dict = {}

    model, program, model_info = create_model(model, optimizer, architecture, model_name)
    

    # training of models
    losses = train_model(program, training_iterations)
    program = create_program(model)
    runner = jax.jit(program.create_runner())
    scores = comparison(model, program, runner, model_info)

    model_dict[model_name] = (model, losses)

    return scores, model_dict






def adata_pca (adata):
    pca_amount = 100
    cells = la.Dim(adata.obs.index, name = "cell")
    pca = sklearn.decomposition.PCA(n_components=pca_amount)
    X = adata.X
    shift = X.mean(0)
    X = X - shift
    pca.fit(np.array(X))
    pca.shift = shift
    return pca,cells,X


# In[88]:


def biased_gene_score (adata,cells, gene_name):
    symbols_gene = eyck.m.t.correlate_single_gene(
        adata, gene_name
    ).index[:10]

    sc.tl.score_genes(
        adata,
        symbols_gene,
        score_name=gene_name+"_bias",
    )
    gene_bias = (
        adata.obs[gene_name+"_bias"]
        - adata.obs[gene_name+"_bias"].mean()
    ) / adata.obs[gene_name+"_bias"].std()
    loc_bias = la.Fixed(
        gene_bias.values,
        definition=la.Definition([cells]),
        label=f"LocBiasGene{gene_name}",
    )
    return loc_bias
    



def biased_initialization(x, loc_bias):
    x_new = la.models.reinitialize(x, generator.key())
    x_new.q.loc = la.links.scalar.Linear(x_new.q.loc, b=loc_bias)
    return x_new


def latent_x(pca, cells,X):
    x = la.Latent(
        la.distributions.Uniform(key=generator.key()),
        definition=la.Definition([cells]),
        label="x",
        key=generator.key(),
    )
    amortization_input = la.Fixed(
        pca.transform(np.array(X))[:, :pca.n_components],
        definition=la.Definition([cells, la.Dim(pca.n_components, name="pca")]),
        label="AmortizationInput",
    )
    n_hidden_dimensions = 100
    encoder = la.amortization.Encoder(
        amortization_input,
        x,
        key = generator.key(),
        n_hidden_dimensions=n_hidden_dimensions,
        n_layers=2,
    )
    return x


def pre_caluclate_pca_biases(pca,cells,X):
    x_bias_loc_list = []
    for pc_i in range(10):
        x_bias = (pca.transform(np.array(X))[:, pc_i] - pca.transform(np.array(X))[:, pc_i].mean())/ pca.transform(np.array(X))[:, pc_i].std()
        loc_bias = la.Fixed(
            x_bias,
            definition=la.Definition([cells]),
            label=f"LocBiasPC{pc_i}",
        )
        x_bias_loc_list.append(loc_bias)
    return x_bias_loc_list


def semi_biased_initialization(x,x_bias_loc_list,pc_i):
    x_new = la.models.reinitialize(x, generator.key())
    x_new.q.loc = la.links.scalar.Linear(x_new.q.loc, b=x_bias_loc_list[pc_i])
    return x_new



def unbiased_initialization(x):
    x_new = la.models.reinitialize(x, generator.key())
    # x.q.loc remains unchanged
    return x_new



def change_model_amortized(model, x, effect_type:str, effect_name:str=None):
    a = la.Latent(p=la.distributions.Normal(key=generator.key(), loc=1, scale=1),label="a",definition=[model.clean[1]])
    knots = la.Dim(10, name="knot")
    knot_positions = la.Fixed(
        np.linspace(0.0, 1.0, len(knots)), definition=[knots], label="knot_positions"
    )
    standard_values = {"Linear":{"x":x,"a":a,},
                    "Spline":{"x":x,"a":True, 'knot':knot_positions},
                    "CircularSpline":{"x":x,"a":True, 'knot':knot_positions},
                    "Sigmoid": {"x":x,"a":a},
                    "Logistic":{"x":x,"a":a}}
    
    parameters = standard_values
    if not effect_name:
        effect_name = effect_type 
    function = getattr(la.links.scalar, effect_type)
    x_effect = function(**parameters[effect_type],label=effect_name,definition=[model.clean[0], model.clean[1]])
    lfc = model.find("lfc")
    setattr(lfc, effect_name, x_effect)
    return model



def change_model_amortized_tuple(model, effects,initialization_func, placeholder=None):
    permutation_list = np.random.permutation(10)

    for i, effect in enumerate(effects):
        effect_name = effect
        if initialization_func.__name__ == 'semi_biased_initialization':
            placeholder[0] = permutation_list[i]
            effect_name_number = permutation_list[i]
        else:
            effect_name_number = i
        x_new = initialization_func()
        effect_name += str(effect_name_number)

        model = change_model_amortized(model,x_new, effect,effect_name)
    return model


def extract_and_crossover(dict1, dict2, crossover_amount):
    # Step 1: Extract numbers from keys and create new dictionaries
    new_dict1 = {int(''.join(filter(str.isdigit, k))): (k, v) for k, v in dict1.items()}
    new_dict2 = {int(''.join(filter(str.isdigit, k))): (k, v) for k, v in dict2.items()}

    # Step 2: Find common and unique keys
    keys_in_common = list(set(new_dict1.keys()) & set(new_dict2.keys()))
    other_keys_dict1 = list(set(new_dict1.keys()) - set(new_dict2.keys()))
    other_keys_dict2 = list(set(new_dict2.keys()) - set(new_dict1.keys()))
    # Step 3: Perform crossover

    for _ in range(crossover_amount):
        if not keys_in_common:
                # Crossover in unique keys
                key1 = random.choice(other_keys_dict1)
                key2 = random.choice(other_keys_dict2)
                new_dict1[key1], new_dict2[key2] = new_dict2[key2], new_dict1[key1]
                other_keys_dict1.remove(key1)  # Remove the key from dict1
                other_keys_dict2.remove(key2)  # Remove the key from dict2   
        elif not other_keys_dict1 or not other_keys_dict2:
                # Crossover in common keys
                key = random.choice(keys_in_common)
                new_dict1[key], new_dict2[key] = new_dict2[key], new_dict1[key]
                keys_in_common.remove(key)  # Remove the key after crossover                     
        else:
            if random.choice([True, False]):  # Randomly decide between common or unique key crossover
                # Crossover in common keys
                key = random.choice(keys_in_common)
                new_dict1[key], new_dict2[key] = new_dict2[key], new_dict1[key]
                keys_in_common.remove(key)  # Remove the key after crossover
            else:
                # Crossover in unique keys
                key1 = random.choice(other_keys_dict1)
                key2 = random.choice(other_keys_dict2)
                new_dict1[key1], new_dict2[key2] = new_dict2[key2], new_dict1[key1]
                other_keys_dict1.remove(key1)  # Remove the key from dict1
                other_keys_dict2.remove(key2)  # Remove the key from dict2

    # Step 4: Rebuild original dictionaries with updated values
    updated_dict1 = {original_key: value for key, (original_key, value) in new_dict1.items()}
    updated_dict2 = {original_key: value for key, (original_key, value) in new_dict2.items()}

    return updated_dict1, updated_dict2



def crossover_dicts(model1, model2):
    lfc1, lfc2 =model1.find("lfc"), model2.find("lfc")
    dict1, dict2 = lfc1.components, lfc2.components
    
    # Get keys from both dictionaries
    items1 = dict1.copy()
    items2 = dict2.copy()
    if not items1.keys() or not items2.keys():
        return
    for key, value in items1.items():
        lfc1.remove_component(key)
    for key, value in items2.items():
        lfc2.remove_component(key)
    crossover_amount = random.randint(1, min(len(items1), len(items2)) - 1)
    updated_dict1, updated_dict2 = extract_and_crossover(items1, items2, crossover_amount)

    for key, value in updated_dict1.items():
        lfc1.add_component(key, value.clone())
    for key, value in updated_dict2.items():
        lfc2.add_component(key, value.clone())
    


# In[99]:


def safe_pickle(data, filename="models.pkl", folder="temporary_safe"):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    temp_filename = "temp"+filename
    # Set up paths for the main and temporary files
    filepath = os.path.join(folder, filename)
    temp_filepath = os.path.join(folder, temp_filename)
    
    try:
        # Save the data to a temporary file
        with open(temp_filepath, 'wb') as f:
            pickle.dump(data, f)
        # Rename the temporary file to the main file
        os.replace(temp_filepath, filepath)
        print(f"Successfully saved to {filepath}")
    except Exception as e:
        print(f"Error during pickling: {e}")
        # If there’s an error, the old file remains intact
    finally:
        # Clean up temp file if it exists and wasn't renamed
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)





def get_architecture_name(model):
    architecture_name = "".join([re.sub(r'\d+', '', arch[:3]) for arch in model.find("lfc").components.keys()])
    if not architecture_name:
        architecture_name = "empty"
    return architecture_name



def latenta_testing_genetic_loop_iteration2(current_subpopulation,optimizer, training_iterations, generation, scores,
                                            initialization_name, model_dict_per_generation, algorithm_iterations,
                                              n_starting_population ,models_ran,models_to_run,lineage, full_crossover):
    
    model_dict = {}
    for i, current_subpopulation_model_dict in enumerate(current_subpopulation):
        for model_name, model in tqdm.tqdm(current_subpopulation_model_dict.items()):
            architecture_name = get_architecture_name(model)
            new_scores, new_model_dict = latenta_processes_basic(model, optimizer, architecture_name, model_name, training_iterations)

            new_scores["generation"] = generation
            new_scores["initialisation"] = initialization_name
            scores = pd.concat([scores, new_scores], axis=0, ignore_index=True)
            model_dict.update(new_model_dict)   
            models_ran +=1
            sys.stdout.write(f"Models trained {models_ran}/{models_to_run}\n")

    model_dict_per_generation[generation] = model_dict
    for key in model_dict.keys():
        ke = re.sub(r'\d+$', '', key)
        lineage[f'gen{generation}_{key}'] = lineage.get(f'gen{generation}_{key}', None)
        lineage[f'gen{generation}_{key}'] = lineage.get(f'gen{generation}_{key}', lineage[f'gen{generation}_{ke+"0"}'])
    if generation == algorithm_iterations-1:
        return current_subpopulation,scores,models_ran

    if full_crossover:
        top_x = n_starting_population
    else:
        top_x = min(n_starting_population//2, 10)
        if top_x%2:
            top_x += 1
    # takes the model names of the top x best models from the previous generation
    selected_for_crossover = list(scores[scores["generation"] == generation].sort_values(by="elbo", ascending=False).iloc[:top_x,]["model_name"])

    top_x_models = [(model_name,model_dict_per_generation[generation][model_name][0]) for model_name in selected_for_crossover]

    # Make random pairs for crossover
    random.shuffle(top_x_models)
    random_pairs = [(top_x_models[i], top_x_models[i + 1]) for i in range(0, len(top_x_models), 2)]

    # we prepare the next population
    current_subpopulation = []

    for (model_name1,model1), (model_name2,model2) in random_pairs:
        crossover_dicts(model1, model2)
        architecture_name1 = get_architecture_name(model1)
        architecture_name2 = get_architecture_name(model2)


        new_model_name1 = architecture_name1+"_"+str(max(len(current_subpopulation),0))+"_"+str(0)
        current_subpopulation.append({new_model_name1:model1})
        new_model_name2 = architecture_name2+"_"+str(max(len(current_subpopulation),1))+"_"+str(0)
        current_subpopulation.append({new_model_name2:model2})


        old_model_tuples = (f'gen{generation}_{model_name1}',f'gen{generation}_{model_name2}')

        lineage[f'gen{generation+1}_{new_model_name1}'] = old_model_tuples            
        lineage[f'gen{generation+1}_{new_model_name2}'] = old_model_tuples
    return current_subpopulation,scores,models_ran
    


# In[175]:


def model_testing_genetic_algorithm(base_model, optimizer, architecture="base_model", n_model_repeats=1, training_iterations=800,
                          effects=["Linear", "Spline", "CircularSpline", "Sigmoid"], n_effects_range=(2, 10),
                          n_starting_population=20, algorithm_iterations=2, initialization_funcs:list=[unbiased_initialization], placeholder:list=[0],full_crossover=False, temporary_folder = "temporary_safe"):
    
    
    # Dictionary to store parent-child relationships
    all_model_dict_per_generation = {}
    all_scores = {}
    lineages = {}
    generation = 0
    models_to_run = (algorithm_iterations * n_starting_population * n_model_repeats*len(initialization_funcs))

    models_ran = 0
    sys.stdout.write(f"Models trained {models_ran}/{models_to_run}\n")

    for initialization_func in initialization_funcs:
        all_model_dict_per_generation[initialization_func.__name__] = {}
        all_scores[initialization_func.__name__] = pd.DataFrame()
        lineages[initialization_func.__name__] = {}
    # Generate initial population
    current_population = {}


    for i in range(n_starting_population):
        if n_effects_range.is_integer():
            n_effects = n_effects_range
        else:
            n_effects = random.randrange(n_effects_range[0], n_effects_range[1])
        effects_list = random.choices(effects, k=n_effects)
        for initialization_func in initialization_funcs:
            models_in_the_loop = {}
            for j in range(n_model_repeats):
                new_model = change_model_amortized_tuple(base_model.clone(), effects_list,initialization_func, placeholder)
                architecture_name = get_architecture_name(new_model)
                model_name = architecture_name+"_"+str(i)+"_"+str(j)
                models_in_the_loop[model_name] = new_model
            if initialization_func.__name__ not in current_population:
                current_population[initialization_func.__name__] = []
            # Append the new model to the list
            current_population[initialization_func.__name__].append(models_in_the_loop)
            
            print(initialization_func.__name__ , i)
    # all model architectures are now stored in current_population with 
    # E.g current_population = {"unbiased_initialization": 
    # [{"SplCirLin_0_0":model1,"SplCirLin_0_1":model2},{"LinLin_1_0":model3,"LinLin_1_1":model4}]}

    # training loop and keeping track of the information
    for generation in range(algorithm_iterations):
        next_population = {}
        for k,current_subpopulation in current_population.items():

            
            model_dict_per_generation = all_model_dict_per_generation[k]
            scores = all_scores[k]
            a_population,a_scores,models_ran = latenta_testing_genetic_loop_iteration2(current_subpopulation,optimizer, training_iterations, generation, scores,
                                            k, model_dict_per_generation, algorithm_iterations,
                                              n_starting_population ,models_ran,models_to_run,lineages[k],full_crossover)

            
            next_population[k] = a_population
            all_scores[k] = a_scores
            data = [all_scores[k], all_model_dict_per_generation[k], lineages[k]]
            names = ["scores", "dictionary", "lineage"]
            for i in range(len(names)):
                safe_pickle(data[i], k+"_"+names[i]+".pkl", temporary_folder)

        if not full_crossover:
            start_value = len(next_population[k])
            end_value = n_starting_population
            for i in range(start_value,end_value):
                print(f"yes"+str({i}))
                n_effects = random.randrange(n_effects_range[0], n_effects_range[1])
                effects_list = random.choices(effects, k=n_effects)            
                for initialization_func in initialization_funcs:
                    models_in_the_loop = {}
                    for j in range(n_model_repeats):
                        new_model = change_model_amortized_tuple(base_model.clone(), effects_list,initialization_func, placeholder)
                        architecture_name = get_architecture_name(new_model)
                        model_name = architecture_name+"_"+str(i)+"_"+str(j)
                        models_in_the_loop[model_name] = new_model

                # Append the new model to the list
                next_population[initialization_func.__name__].append(models_in_the_loop)

        current_population = next_population.copy()


    scores = pd.concat([scores for scores in all_scores.values()], axis=0, ignore_index=True)
    scores["relative_elbo"] = scores["elbo"] - scores["elbo"].min()
    scores["relative_lik"] = scores["lik"] - scores["lik"].min()
    scores["relative_penalty"] = scores["penalty"] - scores["penalty"].min()
    scores = scores.reset_index()


    return scores, all_model_dict_per_generation, lineages


def model_testing_elitism(model, optimizer, architecture, n_model_repeats=1, training_iterations=800,
                          effects=["Linear", "Spline", "CircularSpline", "Sigmoid"], algorithm_iterations=2,initialization_funcs:list=[unbiased_initialization], placeholder:list=[0]):
    # Dictionary to store parent-child relationships
    all_model_dict_per_generation = {}
    all_scores = {}
    lineages = {}
    # i need this dictionary for the loop so i have a best model per initialization method
    best_model = {}
    
    generation = 0
    models_to_run = (algorithm_iterations * n_model_repeats * len(effects) * len(initialization_funcs))
    new_scores, new_model_dict = latenta_processes_basic(model, optimizer, architecture, architecture+'_0_0', 1000)
    models_ran = 0
    sys.stdout.write(f"Models trained {models_ran}/{models_to_run}\n")
    for initialization_func in initialization_funcs:
        all_model_dict_per_generation[initialization_func.__name__] = {}
        all_scores[initialization_func.__name__] = pd.DataFrame()
        lineages[initialization_func.__name__] = {}
        best_model[initialization_func.__name__] = new_model_dict[architecture+'_0_0'][0].clone()

    for generation in range(algorithm_iterations):
        # we give the first pc in generation 1, second pc in generation 2 etc.
        placeholder[0] = generation
        for initialization_func in initialization_funcs:
            k = initialization_func.__name__
            scores = all_scores[k]
            model_dict_per_generation = all_model_dict_per_generation[k]
            lineage = lineages[k]
            model_dict_per_generation[generation] = {}
            model_dict = model_dict_per_generation[generation]
            for i,effect in enumerate(effects):
                for j in range(n_model_repeats):
                    #create new model
                    new_model = best_model[initialization_func.__name__].clone()
                    x = initialization_func()
                    new_model = change_model_amortized(new_model,x, effect,effect+str(generation))
                    # get architecture name
                    architecture_name = get_architecture_name(new_model)
                    model_name = architecture_name+"_"+str(i)+"_"+str(j)

                    # run training and create pd scores and model dict
                    new_scores, new_model_dict = latenta_processes_basic(new_model, optimizer, architecture_name, model_name, training_iterations)
                    # print progress in terminal
                    models_ran += 1
                    sys.stdout.write(f"Models trained {models_ran}/{models_to_run}\n")

                    new_scores["generation"] = generation
                    new_scores["initialisation"] = k
                    scores = pd.concat([scores, new_scores], axis=0, ignore_index=True)
                    model_dict.update(new_model_dict)
            model_dict_per_generation[generation] = model_dict
            all_scores[k] = scores
            for key in model_dict.keys():
                try:
                    lineage[f'gen{generation}_{key}'] = [f'gen{max(0,generation-1)}_{best_model_name}']
                except:
                    continue
            # Filter rows with generation == 0
            generation_scores = scores[scores["generation"] == generation]

            # Get the row with the maximum 'elbo' value in generation 0
            best_model_name = generation_scores.loc[generation_scores['elbo'].idxmax()]["model_name"]
            print(best_model_name)
            best_model[initialization_func.__name__] = model_dict[best_model_name][0]
            
            for key in model_dict.keys():
                lineage[f'gen{generation}_{key}'] = lineage.get(f'gen{generation}_{key}', None)
                key_no_digits = re.sub(r'\d+$', '', key)
                lineage[f'gen{generation}_{key}'] = lineage.get(f'gen{generation}_{key}', lineage[f'gen{generation}_{key_no_digits+"0"}'])

            model_dict = {}
            

            data = [all_scores[k], all_model_dict_per_generation[k], lineages[k]]
            names = ["scores", "dictionary", "lineage"]
            for i in range(len(names)):
                safe_pickle(data[i], k+"_"+names[i]+".pkl", folder="temporary_safe")
    
    scores = pd.concat([scores for scores in all_scores.values()], axis=0, ignore_index=True)
    scores["relative_elbo"] = scores["elbo"] - scores["elbo"].min()
    scores["relative_lik"] = scores["lik"] - scores["lik"].min()
    scores["relative_penalty"] = scores["penalty"] - scores["penalty"].min()
    scores = scores.reset_index()
    



    return scores,all_model_dict_per_generation,lineages


def initialization_func_generator(initialization_type, adata):
    pca,cells,X =adata_pca(adata)
    x = latent_x(pca,cells,X)
    if initialization_type == 'unbiased_initialization':
        initialization_func1 = lambda: unbiased_initialization(x)
        initialization_func1.__name__ = 'unbiased_initialization'
        return initialization_func1
    elif initialization_type == 'semi_biased_initialization':
        x_bias_loc_list = pre_caluclate_pca_biases(pca,cells,X)
        placeholder = [0]
        initialization_func2 = lambda: semi_biased_initialization(x,x_bias_loc_list, placeholder[0])
        initialization_func2.__name__ = 'semi_biased_initialization'
        return initialization_func2

def relative_values_dataframe (scores):
    try:
        scores["relative_elbo"] = scores["elbo"] - scores["elbo"].min()
        scores["relative_lik"] = scores["lik"] - scores["lik"].min()
        scores["relative_penalty"] = scores["penalty"] - scores["penalty"].min()
        scores = scores.reset_index(drop=True)
    except:
        pass
    return scores