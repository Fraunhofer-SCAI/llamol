from typing import Dict, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

from rdkit.Chem import AllChem, Descriptors, RDConfig

import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
# now you can import sascore!
import sascorer
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)
plt.rcParams.update({'font.size': 13.0})
# plt.rcParams.update({"font.size": 12.5})

COL_TO_DISPLAY_NAME = {
    "logp": "LogP",
    "sascore": "SAScore",
    "mol_weight": "Molecular Weight / 100",
}


def calcContextSAScore(smiles: List[str]):
    sasc = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        sa = sascorer.calculateScore(mol)
        sasc.append(sa)

    return np.array(sasc)


def calcContextLogP(smiles: List[str]):
    logps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        logp = Descriptors.MolLogP(mol)
        logps.append(logp)

    return np.array(logps)


def calcContextEnergy(smiles, num_confs=5):
    contexts = []
    for smi in smiles:
        # print("Calculating Energy:",smi)
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMultipleConfs(mol, num_confs, numThreads=48)
        generated_smiles = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=48)
        energies = []
        for coverged, energy in generated_smiles:
            if coverged != 0:
                print("Not converged!", smi)
            energies.append(energy)

        # print(energy)
        # kcal/mol
        mean_en = np.mean(energies)
        # to hartree
        mean_en = mean_en * 0.0016
        contexts.append(mean_en)

    return np.array(contexts)


def calcContextMolWeight(smiles: List[str]):
    con = []
    for _, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        c = Descriptors.ExactMolWt(mol) / 100
        con.append(c)

    return np.array(con)


def plot_1D_condition(
    context_col,
    save_path,
    new_context,
    generated_smiles,
    temperature,
    context_dict,
    context_scaler=None,
    fontsize = 15
):
    for con_idx, con_col in enumerate(context_col):
        save_path = os.path.join(
            save_path, f"{con_col}_{'-'.join(context_col)}_temp{temperature}"
        )
        os.makedirs(save_path, exist_ok=True)

        current_context = new_context[con_col].cpu().detach().numpy()
        if con_col == "mol_weight":
            predicted_context = calcContextMolWeight(generated_smiles)
        elif con_col == "logp":
            predicted_context = calcContextLogP(generated_smiles)
        elif con_col == "sascore":
            predicted_context = calcContextSAScore(generated_smiles)
        elif con_col == "energy":
            # TODO: Change to something better
            predicted_context = calcContextEnergy(generated_smiles)

        if context_scaler is not None:
            raise NotImplementedError("Not implemented yet")
            # context_list = context_scaler.inverse_transform(context_list)

        mean_vals_pred = []
        labels = np.unique(current_context)
        mse_value = []
        mad_value = []
        for label in labels:
            mask = (current_context == label).reshape(-1)
            mean_val = np.mean(predicted_context[mask])
            mean_vals_pred.append(mean_val)
            mse_value.extend((predicted_context[mask] - label) ** 2)
            mad_value.extend(abs(predicted_context[mask] - label))

        mse = np.mean(mse_value)
        mad = np.mean(mad_value)
        logger.info(f"MSE {mse}")
        logger.info(f"MAD {mad}")
        logger.info(f"SD: {np.std(mad_value)}")

        current_context = current_context.reshape(-1)

        # Create a figure and axes
        fig, ax1 = plt.subplots()

        # Scatter plot
        # ax1.scatter(
        #     current_context,
        #     predicted_context,
        #     label="Ground Truth vs Prediction",
        #     c="blue",
        #     alpha=0.5,
        # )
        print(con_idx)
        ax1 = sns.violinplot(
            x=current_context.astype(float),
            y=predicted_context.astype(float),
            native_scale=True,
            ax=ax1,
            orient="x",
            inner="box",
            label="Ground Truth vs Prediction" if con_idx == 0 else None,
            legend="brief",
            color="blue",
            alpha=0.7,
        )
        ax1.plot(
            np.arange(np.min(current_context), np.max(current_context) + 1),
            np.arange(np.min(current_context), np.max(current_context) + 1),
            label="y=x",
            c="black",
        )

        ax1.scatter(labels, mean_vals_pred, label="Mean predicted values", c="red")
        ax1.set_xlabel("Ground Truth",fontsize = fontsize)
        ax1.set_ylabel("Prediction",fontsize = fontsize)

        # Histogram
        ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis
        
        # ax2.set_yscale("log")
        sns.histplot(
            context_dict[con_col],
            # bins=200,
            label="Dataset distribution",
            alpha=0.5,
            # kde=True,
            # element="poly",
            ax=ax2,
        )
        # ax2.hist(
        #     context_dict[con_col],
        #     bins=200,
        #     label="Dataset distribution",
        #     alpha=0.5,
        # )
        ax2.set_ylabel("Frequency", fontsize = fontsize)

        # Combine legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        all_handles = handles1 + handles2
        all_labels = labels1 + labels2

        unique_labels = []
        unique_handles = []
        for handle, label in zip(all_handles, all_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        ax1.legend(unique_handles, unique_labels,  prop={'size': fontsize - 5})
        ax1.tick_params(axis='both', which='major', labelsize=fontsize - 5)
        # handles1, labels1 = ax1.get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()

        # ax1.legend(handles1 + handles2, labels1 + labels2)

        plt.xlim((np.min(current_context) - 1, np.max(current_context) + 1))
        # Set title
        display_name = COL_TO_DISPLAY_NAME[con_col]
        plt.title(f"{display_name} - temperature: {temperature:.2f} \nmad: {round(mad, 4):.4f}",fontsize = fontsize)
        plt.subplots_adjust(right=0.8)
        out_df = pd.DataFrame(
            {
                "smiles": generated_smiles,
                f"{con_col}": predicted_context.tolist(),
                f"target_{con_col}": current_context.tolist(),
            }
        )
        out_df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)
        out_path = os.path.join(save_path, "graph.png")
        print(f"Saved to {out_path}")
        plt.savefig(out_path)
        plt.clf()


def plot_2D_condition(
    context_col,
    save_path,
    new_context,
    generated_smiles,
    temperature,
    label: Union[str, None] = None,
):
    save_path = os.path.join(
        save_path, f"multicond2_{'-'.join(context_col)}_temp={temperature}"
    )
    if label is not None:
        save_path = os.path.join(save_path, label)

    os.makedirs(save_path, exist_ok=True)
    delta_dict = {c: [] for c in context_col}
    predicted_context_dict = {}
    for con_col in context_col:
        current_context = new_context[con_col].cpu().numpy()
        if con_col == "mol_weight":
            predicted_context = calcContextMolWeight(generated_smiles)
        elif con_col == "logp":
            predicted_context = calcContextLogP(generated_smiles)
        elif con_col == "sascore":
            predicted_context = calcContextSAScore(generated_smiles)
        elif con_col == "energy":
            # TODO: Change to something better
            predicted_context = calcContextEnergy(generated_smiles)

        predicted_context_dict[con_col] = np.array(predicted_context)
        delta_dict[con_col] = np.abs(current_context - np.array(predicted_context))

        # Create a DataFrame from delta_dict
    df = pd.DataFrame(delta_dict)
    real_values_prop1 = new_context[context_col[0]].cpu().numpy()
    real_values_prop2 = new_context[context_col[1]].cpu().numpy()
    # cmap = plt.get_cmap('Blues')  # Choose a green color palette from Matplotlib
    mse_vals_x = []
    mad_vals_x = []
    mse_vals_y = []
    mad_vals_y = []
    fig = plt.figure()
    ax = plt.subplot(111)
    for v1 in np.unique(real_values_prop1):
        for v2 in np.unique(real_values_prop2):
            mask = (real_values_prop1 == v1) & (real_values_prop2 == v2)
            indices = np.nonzero(mask)[0]
            # print("Indices", len(indices))
            # Get the color from the color palette based on the v1 value
            # color = cmap((v1 - np.min(real_values_prop1)) / (np.max(real_values_prop1) - np.min(real_values_prop1)))
            color = np.random.rand(
                3,
            )
            # # Plot scatter plot with the specified color and label

            x_pred = predicted_context_dict[context_col[0]][indices].ravel()
            y_pred = predicted_context_dict[context_col[1]][indices].ravel()
            mse_vals_x.extend((x_pred - v1) ** 2)
            mad_vals_x.extend(np.abs(x_pred - v1))

            mse_vals_y.extend((y_pred - v2) ** 2)
            mad_vals_y.extend(np.abs(y_pred - v2))

            ax.scatter(x_pred, y_pred, color=color, alpha=0.5)

            # Plot KDE plot with the specified color
            # sns.kdeplot(
            #     data=pd.DataFrame(
            #         {
            #             f"x": x_pred,
            #             f"y": y_pred,
            #         }
            #     ),
            #     x=f"x",
            #     y=f"y",
            #     color=color,
            #     fill=False,
            #     bw_adjust=2.25,
            #     # label=f"({v1}, {v2})"
            # )

            ax.scatter(v1, v2, color=color, label=f"({v1}, {v2})", marker="^", s=20.0)

    mse_x = np.mean(mse_vals_x)
    mad_x = np.mean(mad_vals_x)
    mse_y = np.mean(mse_vals_y)
    mad_y = np.mean(mad_vals_y)

    logger.info(f"MSE {context_col[0]}: {mse_x}")
    logger.info(f"MAD {context_col[0]}: {mad_x}")
    logger.info(f"MSE {context_col[1]}: {mse_y}")
    logger.info(f"MAD {context_col[1]}: {mad_y}")

    file_path = os.path.join(save_path, "metrics.txt")

    with open(file_path, "w") as f:
        f.write(f"MSE {context_col[0]}: {mse_x} \n")
        f.write(f"MAD {context_col[0]}: {mad_x} \n")
        f.write(f"MSE {context_col[1]}: {mse_y} \n")
        f.write(f"MAD {context_col[1]}: {mad_y} \n")

    ax.set_xlabel(COL_TO_DISPLAY_NAME[context_col[0]])
    ax.set_ylabel(COL_TO_DISPLAY_NAME[context_col[1]])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlim((np.min(real_values_prop1) -1, np.max(real_values_prop1)+1))
    ax.set_ylim((np.min(real_values_prop2) -1, np.max(real_values_prop2)+1))
    
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Multi Property Distribution of Generated Molecules")
    out_path = os.path.join(save_path, "graph.png")
    logger.info(f"Saved to {out_path}")
    plt.savefig(out_path)
    plt.clf()
    return save_path


def plot_3D_condition(
    context_col, save_path, new_context, generated_smiles, temperature
):
    save_path = os.path.join(
        save_path, f"multicond3_{'-'.join(context_col)}_temp={temperature}"
    )
    os.makedirs(save_path, exist_ok=True)
    predicted_context_dict = {}
    for con_col in context_col:
        predicted_context = calc_context_from_smiles(generated_smiles, con_col)

        predicted_context_dict[con_col] = np.array(predicted_context)

    real_values_prop1 = new_context[context_col[0]].cpu().numpy()
    real_values_prop2 = new_context[context_col[1]].cpu().numpy()
    real_values_prop3 = new_context[context_col[2]].cpu().numpy()
    # cmap = plt.get_cmap('Blues')  # Choose a green color palette from Matplotlib

    mse_vals_x = []
    mad_vals_x = []
    mse_vals_y = []
    mad_vals_y = []
    mse_vals_z = []
    mad_vals_z = []

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for v1 in np.unique(real_values_prop1):
        for v2 in np.unique(real_values_prop2):
            for v3 in np.unique(real_values_prop3):
                mask = (
                    (real_values_prop1 == v1)
                    & (real_values_prop2 == v2)
                    & (real_values_prop3 == v3)
                )
                indices = np.nonzero(mask)[0]
                # print("Indices", len(indices))
                # Get the color from the color palette based on the v1 value
                # color = cmap((v1 - np.min(real_values_prop1)) / (np.max(real_values_prop1) - np.min(real_values_prop1)))
                color = np.random.rand(
                    3,
                )

                x_pred = predicted_context_dict[context_col[0]][indices].ravel()
                y_pred = predicted_context_dict[context_col[1]][indices].ravel()
                z_pred = predicted_context_dict[context_col[2]][indices].ravel()

                mse_vals_x.extend((x_pred - v1) ** 2)
                mad_vals_x.extend(np.abs(x_pred - v1))

                mse_vals_y.extend((y_pred - v2) ** 2)
                mad_vals_y.extend(np.abs(y_pred - v2))

                mse_vals_z.extend((z_pred - v3) ** 2)
                mad_vals_z.extend(np.abs(z_pred - v3))

                # # Plot scatter plot with the specified color and label
                ax.scatter(v1, v2, v3, color=color, label=f"({v1}, {v2}, {v3})", s=20.0)
                ax.scatter(
                    x_pred,
                    y_pred,
                    z_pred,
                    color=color,
                )

    mse_x = np.mean(mse_vals_x)
    mad_x = np.mean(mad_vals_x)
    mse_y = np.mean(mse_vals_y)
    mad_y = np.mean(mad_vals_y)
    mse_z = np.mean(mse_vals_z)
    mad_z = np.mean(mad_vals_z)

    logger.info(f"MSE {context_col[0]}: {mse_x}")
    logger.info(f"MAD {context_col[0]}: {mad_x}")
    logger.info(f"MSE {context_col[1]}: {mse_y}")
    logger.info(f"MAD {context_col[1]}: {mad_y}")
    logger.info(f"MSE {context_col[2]}: {mse_z}")
    logger.info(f"MAD {context_col[2]}: {mad_z}")

    file_path = os.path.join(save_path, "metrics.txt")

    with open(file_path, "w") as f:
        f.write(f"MSE {context_col[0]}: {mse_x} \n")
        f.write(f"MAD {context_col[0]}: {mad_x} \n")

        f.write(f"MSE {context_col[1]}: {mse_y} \n")
        f.write(f"MAD {context_col[1]}: {mad_y} \n")

        f.write(f"MSE {context_col[2]}: {mse_z} \n")
        f.write(f"MAD {context_col[2]}: {mad_z} \n")

    ax.set_xlabel(COL_TO_DISPLAY_NAME[context_col[0]])
    ax.set_ylabel(COL_TO_DISPLAY_NAME[context_col[1]])
    ax.set_zlabel(COL_TO_DISPLAY_NAME[context_col[2]])
    # plt.legend(
    #     bbox_to_anchor=(1.0, 0.5),
    #     loc="center right",
    #     bbox_transform=plt.gcf().transFigure,
    # )
    # plt.subplots_adjust(left=0.05, bottom=0.1, right=0.8)
    plt.legend(
        bbox_to_anchor=(1.035, 0.5),
        loc="center right",
        bbox_transform=plt.gcf().transFigure,
    )
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.775)

    plt.title("Multi Property Distribution of Generated Molecules")
    out_path = os.path.join(save_path, "graph.png")
    print(f"Saved to {out_path}")
    plt.savefig(out_path)
    plt.clf()

    return save_path


def calc_context_from_smiles(generated_smiles, con_col):
    if con_col == "mol_weight":
        predicted_context = calcContextMolWeight(generated_smiles)
    elif con_col == "logp":
        predicted_context = calcContextLogP(generated_smiles)
    elif con_col == "sascore":
        predicted_context = calcContextSAScore(generated_smiles)
    elif con_col == "energy":
        # TODO: Change to something better
        predicted_context = calcContextEnergy(generated_smiles)
    return predicted_context


def plot_unconditional(
    out_path: str = os.getcwd(),
    smiles: List[str] = [],
    temperature: float = 0.8,
    cmp_context_dict: Union[Dict[str, np.array], None] = None,
    context_cols: List[str] = ["logp", "sascore", "mol_weight"],
    fontsize = 17
):
    out_path = os.path.join(out_path, "unconditional")
    os.makedirs(out_path, exist_ok=True)
    sns.set(font_scale=1.25)
    for c in context_cols:
        plt.clf()

        context_cal = calc_context_from_smiles(smiles, c)

        if cmp_context_dict is not None:
            sns.histplot(
                cmp_context_dict[c],
                stat="density",
                label="Dataset Distribution",
                alpha=0.75,
                color="blue",
                bins=500
            )
        sns.histplot(
            context_cal,
            stat="density",
            label="Generated Molecules Distribution",
            alpha=0.5,
            color="orange",
            bins=500
        )

        if c == "logp":
            plt.xlim((-6, 8))
        else:
            plt.xlim((0, 10))

        plt.xlabel(COL_TO_DISPLAY_NAME[c],fontsize=fontsize - 3)
        plt.title(
            f"Unconditional Distribution {COL_TO_DISPLAY_NAME[c]} \nwith Temperature {temperature}", fontsize=fontsize
        )
        plt.legend()

        out_file = os.path.join(out_path, f"unc_{c}_temp={temperature}.png")
        plt.savefig(out_file)
        logger.info(f"Saved Unconditional to {out_file}")


def novelty(gen, train):
    gen_smiles_set = set(gen) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)


def unique_at(gen, k=1000):
    gen = gen[:k]

    return len(set(gen)) / len(gen)


def check_metrics(generated_smiles: List[str], dataset_smiles: List[str]):
    len_before = len(generated_smiles)
    generated_smiles = [g for g in generated_smiles if g is not None]
    len_after = len(generated_smiles)
    unique_gen_smiles = set(generated_smiles)
    novel = novelty(unique_gen_smiles, dataset_smiles)
    unique_at_1k = unique_at(generated_smiles, k=1000)
    unique_at_10k = unique_at(generated_smiles, k=10000)
    return dict(
        novelty=novel,
        unique_at_1k=unique_at_1k,
        unique_at_10k=unique_at_10k,
        validity=len_after / float(len_before),
    )
