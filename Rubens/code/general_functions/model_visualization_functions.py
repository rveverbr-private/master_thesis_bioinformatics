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
import textwrap
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
import networkx as nx
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_lineage(lineage, edges_only=False):
    # Create graph visualization of the lineage
    G = nx.DiGraph()
    for child, parents in lineage.items():
        G.add_node(child)
        if parents:
            for parent in parents:
                G.add_edge(parent, child)
    if edges_only:
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]
        H = G.subgraph(nodes_with_edges)  # Create a subgraph with only nodes that have edges
        G = H
    # Set custom attributes for each node (e.g., "generation")
    generation_mapping = {key: int(key.split('_')[0][3:]) for key in lineage.keys()}

    # Add generation attribute to each node in the graph
    for model in G.nodes:
        G.nodes[model]['generation'] = generation_mapping.get(model, 0)  # Default to generation 0 if not found

    # Use a Matplotlib colormap (e.g., 'viridis', 'plasma', 'inferno', etc.)
    cmap = cm.tab20  # Choose a colormap
    norm = Normalize(vmin=0, vmax=max(generation_mapping.values()))  # Normalize generations to fit colormap

    # Get the colors based on the 'generation' attribute
    node_colors = [cmap(norm(G.nodes[node]['generation'])) for node in G.nodes]

    # Custom node positions: Arrange nodes from left to right by generation
    pos = {}
    y_offset = 0.5  # Start the x-coordinate for the first generation
    generation_positions = {}  # Store positions for each generation

    # Assign positions based on generations
    for node in G.nodes:
        gen = G.nodes[node]['generation']
        if gen not in generation_positions:
            generation_positions[gen] = []  # Create an empty list for each generation

        # For each generation, assign a different y-coordinate
        # This will space the nodes vertically within the same generation
        y_pos = len(generation_positions[gen])  # This ensures each node in the same generation gets a different y value
        if gen % 2:

            y_pos += y_offset
        generation_positions[gen].append(y_pos)

        # Set the position for the node (x is based on generation, y is spaced vertically)
        pos[node] = (gen, y_pos)



    # Draw the graph with custom node positions
    plt.figure(figsize=(20, 10))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=10, font_weight="bold", arrows=True)
    plt.title("Model Lineage with Structured Node Layout")
    plt.show()

def recursive_traversal(obj, element_dict: dict):
    # Base case: if the object has no components, stop the recursion
    if obj.label == "q":
        try:
            definition = obj.value_definition[0].name
            if definition in ["cell", "gene"]:
                # Use the last two parts of the UUID as the key
                element_dict[".".join(obj.uuid.split('.')[-3:])] = (obj, definition)
        except Exception as e:
            pass  # Ignore exceptions and continue
        return  # Base case exit

    # Loop through the components of the current object
    for component in obj.components:
        # Get the parent object for the current component
        parent = getattr(obj, component, None)

        # Recursively traverse the next parent object
        recursive_traversal(parent, element_dict)


def basic_data_adata_converter (model, complete=False, size=1000):
    model.rootify()
    observation = model.p
    k = la.utils.key.StatefulKey(0)
    minibatcher_validation = la.train.minibatching.Minibatcher(model.clean[0], k(), size=size,permute=False, complete=complete)
    minibatcher_validation.initialize()
    program = la.programs.Inference(root = model, minibatcher=minibatcher_validation)
    observation.value(program)
    # create dict with parameters x and a to make sure the program keeps track of their values.

    
    outputs = program.run_all(n=10)
    arrays = [output[("root.p", "value")] for output in outputs]

    # Stack arrays along a new axis and compute the mean
    observation_value = np.mean(np.stack(arrays), axis=0)
    adata = sc.AnnData(
    observation_value)
    adata.layers["counts"] = observation_value.copy()

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata



def data_adata_converter (model, complete=False, size=1000, normalization=False):
    model.rootify()
    observation = model

    k = la.utils.key.StatefulKey(0)
    minibatcher_validation = la.train.minibatching.Minibatcher(model.clean[0], k(), size=size,permute=False, complete=complete)
    minibatcher_validation.initialize()
    program = la.programs.Inference(root = model, minibatcher=minibatcher_validation)
    observation.value(program)
    # create dict with parameters x and a to make sure the program keeps track of their values.
    element_dict = {}
    recursive_traversal(observation, element_dict)
    for element_name, [element, definition_name] in element_dict.items():
        element.value(program)

    
    outputs = program.run_all(n=1)
    arrays = np.concat([output[("root", "value")] for output in outputs])
    # Stack arrays along a new axis and compute the mean
    observation_value = arrays
    adata = sc.AnnData(
    observation_value,
    )
    for element_name, [element, definition_name] in element_dict.items():
        
        if definition_name == "cell":
            arrays = np.concat([output[(element.uuid, "value")] for output in outputs])
            element_value = arrays
            adata.obs[element_name] = element_value
        else:
            try:
                arrays = np.concat([output[(element.uuid, "value")] for output in outputs])
                element_value = arrays
                adata.var[element_name] = element_value
            except:
                pass
    adata.layers["counts"] = observation_value.copy()
    if normalization:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata

def best_models(all_model_dict_per_generation, scores):
    all_best_models_per_generation = {}
    all_best_model_scores = {}
    for k in all_model_dict_per_generation.keys():
        best_models_per_generation = []
        best_model_scores = []
        model_dict_per_generation = all_model_dict_per_generation[k]
        selected_scores = scores[scores["initialisation"] == k]
        for i in range(int(selected_scores["generation"].max())+1):
            generation_df = selected_scores[selected_scores["generation"] == i]
            best_model_row = generation_df.loc[generation_df['elbo'].idxmax()]
            model_name = best_model_row["model_name"]
            best_model_score = best_model_row["elbo"]
            model_dict = model_dict_per_generation[i]
            best_model = model_dict[model_name][0]
            best_models_per_generation.append(best_model)
            best_model_scores.append(best_model_score)
        all_best_models_per_generation[k] = best_models_per_generation
        all_best_model_scores[k] = best_model_scores
    return all_best_models_per_generation, all_best_model_scores

def create_adata_list (all_best_models_per_generation):
    all_adata_object_lists = {}
    for key in all_best_models_per_generation.keys():
        best_models_per_generation = all_best_models_per_generation[key]
        adata_object_list = []
        
        for i,best_model in enumerate(best_models_per_generation):
            adata = data_adata_converter(best_model)
            adata_object_list.append(adata)
            print(f"completed_{i}")
        all_adata_object_lists[key] = adata_object_list
        print(key)
    return all_adata_object_lists

def plot_adata_generations(all_best_models_per_generation, all_adata_object_lists, all_best_model_scores):

    # Flatten all scores into a single list to compute global normalization
    all_scores = [score for scores in all_best_model_scores.values() for score in scores]

    # Compute global normalization and relative scores
    global_min = min(all_scores)
    global_max = max(all_scores)
    global_norm_scores = (np.array(all_scores) - global_min) / (global_max - global_min)
    global_relative_scores = np.array(all_scores) - global_min

    # Map global scores back to their respective keys
    global_norm_scores_per_key = {}
    global_relative_scores_per_key = {}
    current_index = 0
    for k, scores in all_best_model_scores.items():
        num_scores = len(scores)
        global_norm_scores_per_key[k] = global_norm_scores[current_index:current_index + num_scores]
        global_relative_scores_per_key[k] = global_relative_scores[current_index:current_index + num_scores]
        current_index += num_scores

    # Determine the number of rows (keys) and columns (adata per key)
    num_rows = len(all_best_models_per_generation.keys())
    max_columns = max(len(all_adata_object_lists[k]) for k in all_best_models_per_generation.keys())

    # Set up the figure and axes grid
    fig, axes = plt.subplots(num_rows, max_columns, figsize=(5 * max_columns, 5 * num_rows))

    # Ensure axes is always a 2D array for consistency
    if num_rows == 1:  # Only one row
        axes = np.expand_dims(axes, axis=0)
    if max_columns == 1:  # Only one column
        axes = np.expand_dims(axes, axis=1)

    for row, k in enumerate(all_best_models_per_generation.keys()):
        best_model_scores = all_best_model_scores[k]
        adata_object_list = all_adata_object_lists[k]

        # Use global normalized and relative scores
        norm_scores = global_norm_scores_per_key[k]
        relative_scores = global_relative_scores_per_key[k]

        for col, (adata, norm_score, relative_score) in enumerate(zip(adata_object_list, norm_scores, relative_scores)):
            ax = axes[row][col]  # Access the correct subplot

            sc.pl.umap(adata, ax=ax, show=False)  # Set show=False to prevent individual plots
            ax.set_title(f'Generation {col+1}', fontsize=12)  # Add generation number as title

            # Wrap obs columns text to fit within the max_width
            max_width = 30
            obs_columns = ', '.join([column.split('.')[0] for column in adata.obs.columns])
            wrapped_text = textwrap.fill(f'effects: {obs_columns}', width=max_width)

            # Display wrapped text below each plot
            ax.text(0.5, -0.2, wrapped_text, ha='center', va='top', transform=ax.transAxes, fontsize=8)

            # Add score below the plot, color-coded by score intensity
            color = cm.seismic(norm_score)  # Use seismic colormap for color intensity
            ax.text(0.5, -0.50, f'Normalized Score: {norm_score:.2f}', ha='center', va='top', transform=ax.transAxes, fontsize=10, color=color)
            ax.text(0.5, -0.60, f'Relative Score: {relative_score}', ha='center', va='top', transform=ax.transAxes, fontsize=10)

        # Add title to the right of the row
        row_title = "Unbiased Initialization" if "unbiased" in k else "Semi-Biased Initialization"
        fig.text(
            x=-0.06,  # Position slightly to the right of the plots
            y=1 - (row + 0.3) / num_rows,  # Middle of the row
            s=row_title,
            ha='left',
            va='center',
            fontsize=14,
            fontweight='bold',
            transform=fig.transFigure,
        )

        # Hide unused subplots in the current row
        for col in range(len(adata_object_list), max_columns):
            fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # Leave space for row titles on the right
    plt.show()


def compare_adata_obs(
    adata1, adata2, name1="Adata1", name2="Adata2",
    compact=False, threshold=0.7
):
    """
    Compare all columns in adata1.obs with all columns in adata2.obs using scatterplots.

    Parameters:
    - adata1, adata2: AnnData objects with `.obs` to compare.
    - name1, name2: Names for the AnnData objects, displayed on the plots.
    - compact (bool): If True, only show plots with correlation >= threshold.
    - threshold (float): Minimum absolute correlation to display in compact mode.
    """
    # Get the columns from both adata1.obs and adata2.obs
    columns_adata1 = adata1.obs.columns
    columns_adata2 = adata2.obs.columns

    # Store pairs of columns to plot
    plot_data = []

    for feature1 in columns_adata1:
        for feature2 in columns_adata2:
            # Skip non-numeric columns
            if adata1.obs[feature1].dtype.kind not in 'biufc' or adata2.obs[feature2].dtype.kind not in 'biufc':
                continue

            # Calculate correlation
            corr = adata1.obs[feature1].corr(adata2.obs[feature2])
            # Store plot data if in compact mode and correlation is above threshold
            if not compact or abs(corr) >= threshold:
                plot_data.append({
                    'feature1': feature1,
                    'feature2': feature2,
                    'correlation': corr
                })

    # Exit if no plots to display
    if not plot_data:
        print("No correlations met the threshold.")
        return
    
    if not compact:
        n_rows = len(columns_adata1)
        n_cols = len(columns_adata2)
    else:

        # Determine the layout of the grid
        n_plots = len(plot_data)
        n_cols = min(4, n_plots)  # Limit to 4 columns per row
        n_rows = (n_plots + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # Plot each pair
    for ax, data in zip(axes, plot_data):
        feature1, feature2, corr = data['feature1'], data['feature2'], data['correlation']

        # Set color based on correlation sign
        color = 'red' if abs(corr) >= threshold else 'blue'
        # Scatterplot
        sns.scatterplot(
            x=adata1.obs[feature1],
            y=adata2.obs[feature2],
            ax=ax,
            alpha=0.7, edgecolor=None, color=color
        )

        # Set x-axis and y-axis labels with dataset names
        ax.set_xlabel(f"{name1}: {feature1.split(".")[0]}", fontsize=10)
        ax.set_ylabel(f"{name2}: {feature2.split(".")[0]}", fontsize=10)

        # Set correlation text below the plot
        ax.text(
            0.5, -0.2, f"r = {corr:.2f}",
            ha="center", va="center", transform=ax.transAxes, fontsize=8
        )

    # Hide unused subplots
    for ax in axes[len(plot_data):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Function to count effects
def count_effects(architecture):
    if architecture == "base_model":
        return 0  # Base model always has 3 effects
    return len(architecture) // 3  # Each effect is 3 letters
def boxplot_per_generation(scores, y = "elbo", split_on_column ="initialisation"):
    # Apply the function to create the new column
    scores["number_of_effects"] = scores["architecture"].apply(count_effects)

    colors1 = ["#4C72B0", "#55A868"]
    colors2 = ["#8DA0CB", "#B3DE69"]

    palette = {}
    dot_palette = {}
    # Set figure size
    plt.figure(figsize=(12, 6))
    for i, element in enumerate(scores[split_on_column].unique()):
        palette[element] = colors1[i]
        dot_palette[element] = colors2[i]
    # Create a boxplot grouped by `number_of_effects` and colored by `initialisation`
    sns.boxplot(
        x="number_of_effects",
        y= y,
        hue="initialisation",  # Add color based on initialization
        data=scores,
        palette=palette,  # Custom palette
        showfliers=False,  # Optional: Remove outliers for clarity
    )
    sns.stripplot(data=scores,
                x="number_of_effects",
                y=y,hue="initialisation",
                palette=dot_palette,
                dodge=True,  # Ensure alignment with boxplot
                alpha=0.9,
                ) 

    # Add titles and labels
    plt.title("Elitism: ELBO per generation", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.legend(title="Initialization Type", fontsize=10)  # Add legend
    plt.grid(alpha=0.3)  # Optional: Add grid for better readability
    plt.show()

def plot_multiple_losses_with_completion(loss_list,loss_names=None, tail_size=50, window_size=10, completion=True):
    """
    Plots the loss values and completion percentages for multiple loss objects with an extra y-axis for percentages.

    Parameters:
    - loss_list: List of loss objects (each can be a DataFrame or Series).
    - tail_size: Number of last values to zoom in on for the second plot.
    - window_size: Size of the rolling window for smoothing the completion percentage.
    """
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    num_losses = len(loss_list)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Secondary y-axis handles for each subplot
    ax1_secondary = axes[0].twinx()
    ax2_secondary = axes[1].twinx()

    for idx, loss_obj in enumerate(loss_list):
        # Convert DataFrame row to Series if necessary
        if isinstance(loss_obj, pd.DataFrame):
            losses = pd.Series(loss_obj.iloc[0])  # Assume first row
        elif isinstance(loss_obj, pd.Series):
            losses = loss_obj
        else:
            raise ValueError(f"Unsupported loss type: {type(loss_obj)}")

        if loss_names:

            loss_label = loss_names[idx]
            completion_label = f'{loss_names[idx]}%'
        else:
            loss_label = f'Loss {idx + 1}'
            completion_label = f'{idx + 1}%'
        # Calculate initial and minimal loss
        initial_loss = losses.iloc[0]
        min_loss = losses.min()

        # Calculate completion percentage
        completion_percentage = ((initial_loss - losses) / (initial_loss - min_loss)) * 100
        

        # Plot all losses and completion on the first subplot
        axes[0].plot(losses, label=loss_label, color=colors[idx % len(colors)], alpha=0.7)

        # Plot the tail for both losses and completion on the second subplot
        tail_losses = losses.tail(tail_size)

        axes[1].plot(tail_losses.index, tail_losses, label=loss_label, color=colors[idx % len(colors)], alpha=0.7)
        if completion:
            smoothed_completion = completion_percentage.rolling(window=window_size).mean()
            ax1_secondary.plot(smoothed_completion, linestyle='dotted', label=completion_label, color=colors[idx % len(colors)])
            smoothed_tail_completion = smoothed_completion.tail(tail_size)
            ax2_secondary.plot(smoothed_tail_completion.index, smoothed_tail_completion, linestyle='dotted', label=completion_label, color=colors[idx % len(colors)])

    # Finalize the left plot
    axes[0].set_title('All Loss Values')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.7, 0.65))
    ax1_secondary.set_ylabel('Completion Percentage')
    ax1_secondary.legend(loc="upper left", bbox_to_anchor=(0.8, 0.45))

    # Finalize the right plot
    axes[1].set_title(f'Last {tail_size} Loss Values')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].grid()
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.7, 0.65))
    ax2_secondary.set_ylabel('Completion Percentage')
    ax2_secondary.legend(loc="upper left", bbox_to_anchor=(0.8, 0.45))

    plt.tight_layout()
    plt.show()


def get_best_models_effect_values(best_models_per_generation):
    best_models_effect_values = {}
    for i,best_model in enumerate(best_models_per_generation):
        k = la.utils.key.StatefulKey(0)
        minibatcher_validation = la.train.minibatching.Minibatcher(best_model.clean[0], k(), size=1000)
        minibatcher_validation.initialize()
        program = la.programs.Inference(root = best_model, minibatcher=minibatcher_validation)

        # put all values in program
        best_model.p.value(program)
        best_model.p.mu.value(program)
        
        components = best_model.find("lfc").components.keys()
        for component in components:
            value_func = getattr(getattr(getattr(best_model.p.mu.rho.lfc, component), "value"),"__call__")
            value_x_func = getattr(getattr(getattr(getattr(getattr(best_model.p.mu.rho.lfc, component),"x"),"q"), "value"),"__call__")
            value_a_func = getattr(getattr(getattr(getattr(getattr(best_model.p.mu.rho.lfc, component),"a"),"q"), "value"),"__call__")
            value_func(program)
            value_x_func(program)
            value_a_func(program)
        # get outputs
        new_outputs = program.run_all(n=10)
        observation_values = np.concatenate([output[("root.p", "value")] for output in new_outputs])
        y_value = np.concatenate([output[("root.p.mu", "value")] for output in new_outputs])
        effects_x_value_list = []
        effect_a_value_list = []
        effects_value_list = []
        labels = []
        for component in components:
            effect_x_value = np.concatenate([output[(f'root.p.mu.rho.lfc.{component}.x.q', "value")] for output in new_outputs])
            effect_a_value = np.concatenate([output[(f'root.p.mu.rho.lfc.{component}.a.q', "value")] for output in new_outputs])
            effect_value = np.concatenate([output[(f'root.p.mu.rho.lfc.{component}', "value")] for output in new_outputs])
            label = component
            labels.append(component)
            effects_x_value_list.append(effect_x_value)
            effect_a_value_list.append(effect_a_value)
            effects_value_list.append(effect_value)
        best_models_effect_values[f"generation{i}"] = {"counts":observation_values,"mu":y_value,"effect_x": effects_x_value_list,"effect_a":effect_a_value_list,"effect": effects_value_list, "label":labels}
        
        print(f"completed_{i}")
    return best_models_effect_values

def plot_best_models_effect_values(best_models_effect_values):
    figures = {}  # Store figures for each generation if needed
    ix = 2  # Index for data extraction

    for gen, generation_data in list(best_models_effect_values.items()):
        # Determine the number of effects in the generation
        num_effects = len(generation_data["effect_x"])

        # Create a subplot grid with 1 row for the generation and one column per effect
        fig, axes = plt.subplots(nrows=1, ncols=num_effects, figsize=(6 * num_effects, 6))
        fig.suptitle(f"Plots for {gen}", fontsize=16)

        # Ensure axes is iterable even if there's only one effect (1 column)
        if num_effects == 1:
            axes = [axes]

        # Plot each effect
        for j in range(num_effects):
            ax = axes[j]
            
            # Get data for plotting and check dimensions
            x_data = generation_data["effect_x"][j]
            counts = generation_data["counts"]
            mu = generation_data["mu"]
            effect = generation_data["effect"][j]
            
            y_counts = counts[ix] if counts.ndim == 1 else counts[:, ix]
            y_mu = mu[ix] if mu.ndim == 1 else mu[:, ix]
            y_effect = effect[ix] if effect.ndim == 1 else effect[:, ix]
            
            # Plot data
            ax.scatter(x_data, y_counts, color="blue", label="counts")
            ax.scatter(x_data, y_mu, color="orange", label="mu")
            ax.set_xlabel("x1")
            ax.set_ylabel("y (orange) and counts (blue)")

            # Secondary axis for the effect
            ax2 = ax.twinx()
            ax2.scatter(x_data, y_effect, color="red", label=generation_data["label"][j])
            ax2.set_ylabel(generation_data["label"][j])
            
            # Title and legends
            ax.set_title(generation_data["label"][j])
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")

        # Layout adjustments
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the suptitle
        figures[gen] = fig  # Store figure in dictionary

    plt.show()
