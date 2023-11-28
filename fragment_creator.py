from abc import ABC
from dataclasses import dataclass
from typing import List, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem.Recap import RecapDecompose

import random


@dataclass
class Fragment:
    smiles: Union[str, None]
    tokens: Union[List[int], None]


class BaseFragmentCreator(ABC):
    """
    Is the base class for all fragment creator and does nothing to the smiles
    """

    def __init__(self) -> None:
        pass

    def create_fragment(self, frag: Fragment) -> Fragment:
        return ""


# This is the method used in the paper
class RandomSubsliceFragmentCreator(BaseFragmentCreator):
    def __init__(self, max_fragment_size=50) -> None:
        super().__init__()
        self.max_fragment_size = max_fragment_size

    def create_fragment(self, frag: Fragment) -> Fragment:
        """
        Creates the random sub slice fragments from the tokens
        """
        tokens = frag.tokens

        startIdx = np.random.randint(0, len(tokens) - 1)

        endIdx = np.random.randint(
            startIdx + 1, min(len(tokens), startIdx + self.max_fragment_size)
        )
        return Fragment(smiles=None, tokens=tokens[startIdx:endIdx])


class BricksFragmentCreator(BaseFragmentCreator):
    def __init__(self) -> None:
        super().__init__()

    def create_fragment(self, frag: Fragment) -> Fragment:
        """
        Creates the Bricks fragments and takes one randomly
        """
        smiles = frag.smiles
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""

        res = list(BRICSDecompose(m, minFragmentSize=3))
        # print(res)
        return random.choice(res)


class RecapFragmentCreator(BaseFragmentCreator):
    def __init__(self) -> None:
        super().__init__()

    def create_fragment(self, frag: Fragment) -> Fragment:
        """
        Creates the Recap fragments and takes one randomly
        """
        smiles = frag.smiles
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""

        res = RecapDecompose(m, minFragmentSize=3).GetAllChildren()
        # print(res)
        return random.choice(res)


class MolFragsFragmentCreator(BaseFragmentCreator):
    def __init__(self) -> None:
        super().__init__()

    def create_fragment(self, frag: Fragment) -> Fragment:
        """
        Creates the Bricks fragments and takes one randomly
        """
        smiles = frag.smiles
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""

        res = list(Chem.rdmolops.GetMolFrags(m, asMols=True))
        res = [Chem.MolToSmiles(m) for m in res]
        # print(res)
        return random.choice(res)


def fragment_creator_factory(key: Union[str, None]):
    if key is None:
        return None

    if key == "mol_frags":
        return MolFragsFragmentCreator()
    elif key == "recap":
        return RecapFragmentCreator()
    elif key == "bricks":
        return BricksFragmentCreator()
    elif key == "rss":
        return RandomSubsliceFragmentCreator()
    else:
        raise ValueError(f"Do not have factory for the given key: {key}")


if __name__ == "__main__":
    from tokenizer import SmilesTokenizer

    tokenizer = SmilesTokenizer()

    creator = BricksFragmentCreator()
    # creator = MolFragsFragmentCreator()

    # creator = RecapFragmentCreator()

    frag = creator.create_fragment("CC(=O)NC1=CC=C(C=C1)O")

    print(frag)
    tokens = tokenizer.encode(frag)
    print(tokens)
    print([tokenizer._convert_id_to_token(t) for t in tokens])
