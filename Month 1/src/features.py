from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

BASE_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHBD": rdMolDescriptors.CalcNumHBD,
    "NumHBA": rdMolDescriptors.CalcNumHBA,
    "NumRotBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
}

def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def rdkit_descriptors(smiles: str) -> dict:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    feats = {k: func(mol) for k, func in BASE_DESCRIPTOR_FUNCS.items()}
    return feats

def morgan_fingerprint_bits(smiles: str, radius=2, nBits=2048):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return list(fp)  # list of 0/1
