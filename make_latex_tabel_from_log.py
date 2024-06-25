import re
import os

import json

def convert_string_to_list(string):
    try:
        result = json.loads(string)
        if isinstance(result, list):
            return result
        else:
            return []
    except:
        return []



def get_1D_table(log_data) -> str:
    # Extract the relevant information from the log data
    data = []
    data = []
    for block in log_data.split("\n\n"):
        # Extract the context columns
        context_columns = re.search(r'Context columns: (.*?)\n', block)
        context_columns = context_columns.group(1) if context_columns else None
        if context_columns is not None and context_columns != "None" and "," in context_columns:
        
            print("Skipping", context_columns)
            continue
            # quit()


        # Extract the context SMILES
        context_smiles = re.search(r'Context SMILES: (.*?)\n', block)
        context_smiles = context_smiles.group(1) if context_smiles else None
        
        # Extract the number of valid generated molecules
        num_valid_generated = re.search(r'Number valid generated: (.*?) %', block)
        num_valid_generated = float(num_valid_generated.group(1)) if num_valid_generated else None
        
        # Extract the metrics (MSE, MAD, SD)
        mse = re.search(r'MSE (.*?)\n', block)
        mse = float(mse.group(1)) if mse else None
        
        mad = re.search(r'MAD (.*?)\n', block)
        mad = float(mad.group(1)) if mad else None
        
        sd = re.search(r'SD: (.*?)\n', block)
        sd = float(sd.group(1)) if sd else None
        
        # Extract the context SMILES (SMARTS)
        context_smarts = re.search(r'context_smarts (.*?)\n', block)
        context_smarts = context_smarts.group(1) if context_smarts else None
        
        # Extract the mean similarity
        mean_sim = re.search(r'Mean sim (.*?)\n', block)
        mean_sim = float(mean_sim.group(1)) if mean_sim else None
        
        # Extract the has sub
        has_sub = re.search(r'Has Sub: .*? (\d+\.\d+)', block)
        has_sub = float(has_sub.group(1)) if has_sub else None
        # Extract the metrics dictionary
        metrics = re.search(r'Metrics: ({.*?})', block)
        metrics = eval(metrics.group(1)) if metrics else None
        
        if metrics is not None:
            for k in metrics.keys():
                metrics[k] *= 100
            
            metrics["validity"] = num_valid_generated

        # Create the data dictionary for the block
        block_data = {
            "context_columns": context_columns,
            "context_smiles": context_smiles,
            "num_valid_generated": num_valid_generated,
            "mse": mse,
            "mad": mad,
            "sd": sd,
            "context_smarts": context_smarts,
            "mean_sim": mean_sim,
            "has_sub": has_sub,
            "metrics": metrics
        }
        
        # Append the block data to the list
        data.append(block_data)

    # Group the blocks with the same context smiles
    grouped_data = {}
    for block_data in data:
        context_smiles = block_data["context_smiles"]
        if context_smiles not in grouped_data:
            grouped_data[context_smiles] = []
        grouped_data[context_smiles].append(block_data)

    # # Print the grouped data
    # for context_smiles, blocks in grouped_data.items():
    #     print(f"Context SMILES: {context_smiles}")
    #     for i, block in enumerate(blocks, start=1):
    #         print(f"Block {i}:")
    #         print(block)
    #         print("---")
    #     print("===")

    # Define the table header
    header = r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\caption{Table for comparing metrics on 1000 generated molecules for each context token sequence.}\label{table:metrics_compare_molfragments}" + "\n" + r"\begin{tabular}{||l|p{0.3\textwidth}|p{0.125\textwidth}|p{0.15\textwidth}|p{0.15\textwidth}|p{0.15\textwidth}||}" + "\n" + r" \hline" + "\n" + r" & Token sequence \acrshort{smiles} & Unconditional {\color{cyan} Uniqueness at 1k [\%]} / {\color{orange} SM [\%]} & LogP \{-2, 0, 2\} MAD / {\color{cyan} Uniqueness at 1k [\%]} / {\color{orange} SM [\%]} & SAScore \{2, 3, 4\} MAD / {\color{cyan} Uniqueness at 1k [\%]} / {\color{orange} SM [\%]} & Molecular Weight \{2, 3, 4\} MAD / {\color{cyan} Uniqueness at 1k [\%]} / {\color{orange} SM [\%]}  \\" + "\n" + r" \hline\hline"
    # Define the table rows
    rows = ""
    names = [
        "Benzene",
        "Thiophene",
        "3-Methylthiophene",
        "Ethanol",
        "Acetaldehyde",
        "Aspirin",
        "Paracetamol",
        "Caffeine",
        "Morphine",
        "Ibuprofen"
    ]
    for i, (context_smiles, blocks) in enumerate(grouped_data.items(), start=1):
        try:
            rnd = lambda x : str(round(float(x),2))
            # Create the row
            breakpoint = 18
            if len(context_smiles) > breakpoint:
                context_smiles = context_smiles[:breakpoint] + " " + context_smiles[breakpoint:]
            row = f" {i} & {context_smiles} ({names[i-1]})"
            for block in blocks:
                # print(block)
                metrics = block["metrics"]
                if block["context_columns"] == "None":
                    row += "& \hspace{0px}{{\color{cyan}{"+ rnd(metrics["unique_at_1k"]) +"}} / {\color{orange}{"+ rnd(block["has_sub"])+"}}}"  
                else:
                    row += "& \hspace{0px}{"+ rnd(block["mad"]) +" / {\color{cyan}{"+ rnd(metrics["unique_at_1k"]) +"}} / {\color{orange}{"+ rnd(block["has_sub"])+"}}}"
                    
            
            row += r"\\" + "\n"
            

            rows += row
        except Exception as e:
            break

    # Define the table footer
    footer = r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"

    # Combine the table components
    latex_table = header + "\n" + rows + footer

    # Print the LaTeX table
    # print(latex_table)
    return latex_table


def extract_2D_or_3D_data(log_data) -> str:
    # Extract the relevant information from the log data
    data = []
    for block in log_data.split("\n\n"):
        # Extract the context columns (which can be 2D or 3D)
        context_columns = re.search(r'Context columns: (\[.*?\])', block)
    

        context_columns = eval(context_columns.group(1)) if context_columns is not None else None
        # print(context_columns)
        if context_columns is None or  len(context_columns) <= 1:
            print("Skipping", context_columns)
            continue
        # Extract the Context SMILES
        context_smiles = re.search(r'Context SMILES: (.*?)\n', block)
        context_smiles = context_smiles.group(1) if context_smiles is not None else None
        # Extract the Number valid generated
        num_valid_generated = re.search(r'Number valid generated: (.*?) %', block)
        num_valid_generated = float(num_valid_generated.group(1)) if num_valid_generated else None
        # Extract the MSE, MAD values for each context column
        mse_values = {}
        mad_values = {}
        for column in context_columns:
            mse = re.search(rf'MSE {column}: (.*?)\n', block)
            mse_values[column] = float(mse.group(1)) if mse else None

            mad = re.search(rf'MAD {column}: (.*?)\n', block)
            mad_values[column] = float(mad.group(1)) if mad else None

        # Extract the context SMILES (SMARTS)
        context_smarts = re.search(r'context_smarts (.*?)$', block)
        context_smarts = context_smarts.group(1) if context_smarts else None

        # Extract the mean similarity
        mean_sim = re.search(r'Mean sim (.*?)$', block)
        mean_sim = float(mean_sim.group(1)) if mean_sim else None

        # Extract the has sub
        has_sub = re.search(r'Has Sub: .*? (\d+\.\d+)', block)
        has_sub = float(has_sub.group(1)) if has_sub else None

        # Extract the metrics dictionary
        metrics = re.search(r'Metrics: ({.*?})', block)
        metrics = eval(metrics.group(1)) if metrics else None

        if metrics is not None:
            for k in metrics.keys():
                metrics[k] *= 100
            metrics["validity"] = num_valid_generated

        # Create the data dictionary for the block
        block_data = {
            "context_columns": context_columns,
            "context_smiles": context_smiles,
            "num_valid_generated": num_valid_generated,
            "mse_values": mse_values,
            "mad_values": mad_values,
            "context_smarts": context_smarts,
            "mean_sim": mean_sim,
            "has_sub": has_sub,
            "metrics": metrics
        }

        # Append the block data to the list
        data.append(block_data)
    
    return data

def create_latex_table_from_data(data):
    # Define the table header
    header = r"\begin{table}[h]" + "\n" + \
             r"\centering" + "\n" + \
             r"\caption{Table for comparing multiple property conditions for 1000 generated molecules using example token sequences.}" + "\n" + \
             r"\label{table:metrics_multicond_compare_molfragments}" + "\n" + \
             r"\begin{tabular}{||p{0.35\textwidth} p{0.075\textwidth} p{0.10\textwidth} p{0.10\textwidth} p{0.10\textwidth} p{0.10\textwidth}||}" + "\n" + \
             r"\hline" + "\n" + \
             r"Token Sequence SMILES & SM[\%] & Uniqueness at 1k [\%] & LogP \{-2, 0, 2\} MAD & SAScore \{2, 3, 4\} MAD & Molecular Weight \{2, 3, 4\} MAD \\" + "\n" + \
             r"\hline\hline"

    # Define the table rows
    rows = ""
    seen = set()
    seen_iter = 0
    names = [
        "Thiophene",
        "Acetaldehyde",
        "Paracetamol",
        "Caffeine",
    ]

    for block_data in data:
        # Token Sequence SMILES
        context_smiles = block_data["context_smiles"]

        # SM[%] is the mean similarity
        has_sub = f"{block_data['has_sub']:.2f}" if block_data['has_sub'] else ""

        # Uniqueness at 1k [%] is part of the metrics
        uniqueness_1k = f"{block_data['metrics']['unique_at_1k']:.2f}" if block_data['metrics'] and 'unique_at_1k' in block_data['metrics'] else ""

        # MAD values for LogP, SAScore, and Molecular Weight
        logp_mad = f"{block_data['mad_values']['logp']:.2f}" if 'logp' in block_data['mad_values'] else ""
        sascore_mad = f"{block_data['mad_values']['sascore']:.2f}" if 'sascore' in block_data['mad_values'] else ""
        mol_weight_mad = f"{block_data['mad_values']['mol_weight']:.2f}" if 'mol_weight' in block_data['mad_values'] else ""
        # if len(context_smiles) > 16:
        #     context_smiles = context_smiles[:16] + " " + context_smiles[16:]

        if context_smiles in seen:
            context_smiles = ""
        else:
            print(seen_iter,seen,context_smiles)
            seen.add(context_smiles)

            context_smiles += f" ({names[seen_iter]})"

            seen_iter += 1
            if seen_iter != 1:
                rows += r"\hline"+"\n"
        # Add the row for the current block
        rows += f"{context_smiles} & {has_sub} & {uniqueness_1k} & {logp_mad} & {sascore_mad} & {mol_weight_mad} \\\\\n"
    # Define the table footer
    footer = r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"

    # Combine the table components
    latex_table = header + "\n" + rows + footer

    return latex_table

def get_2D_or_3D_table(log_data : str) -> str:
    data = extract_2D_or_3D_data(log_data)
    table = create_latex_table_from_data(data)
    return table


if __name__ == "__main__":
        
    with open(os.path.join(os.path.dirname(__file__), "new_gen_log_FULL_can_canmodel_tokensseq.out"), "r") as f:
        log_data = f.read()
    
    table = get_1D_table(log_data)
    print(table)

    with open(os.path.join(os.path.dirname(__file__), "new_gen_log_FULL_can_canmodel_tokensseq.out"), "r") as f:
        log_data = f.read()

    # print(get_2D_or_3D_table(log_data))