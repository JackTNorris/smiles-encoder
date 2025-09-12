from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
from chemprop import models, featurizers, data
import torch
from typing import Literal
from functools import wraps
from huggingface_hub import hf_hub_download
import numpy as np
from numpy.typing import NDArray


# decorator for verifying smiles input
def verify_smiles(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to get the "smiles" argument by name
        if "smiles" in kwargs:
            value = kwargs["smiles"]
        else:
            # checking positional
            func_args = func.__code__.co_varnames
            if "smiles" in func_args:
                index = func_args.index("smiles")
                value = args[index]
            else:
                raise ValueError("Function does not accept a 'smiles' argument")
        if Chem.MolFromSmiles(value) == None:
            raise ValueError(f"Invalid smiles: {value!r}, RDKit Mol doesn't exist")

        return func(*args, **kwargs)
    return wrapper

class SMILESEncoder():
    def __init__(self, tensor_map_location='cpu'):

        # setting tensor map location
        self._tensor_map_location = tensor_map_location

        # specific chemberta model
        chemberta_model_name = "seyonec/ChemBERTa-zinc-base-v1"

        # morgan generators
        self._morgan_gens = {
            "ecfp4": Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=Chem.rdFingerprintGenerator.GetMorganAtomInvGen()),
            "ecfp6": Chem.rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, atomInvariantsGenerator=Chem.rdFingerprintGenerator.GetMorganAtomInvGen()),
            "fcfp4": Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=Chem.rdFingerprintGenerator.GetMorganFeatureAtomInvGen()),
            "fcfp6": Chem.rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, atomInvariantsGenerator=Chem.rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
        }
        # RDKit fingerprint
        self._rdk_gen = Chem.rdFingerprintGenerator.GetRDKitFPGenerator()
        # featurizer used by stokes
        self._rdk_stokes_featurizer = featurizers.V1RDKit2DNormalizedFeaturizer()
        # chemberta model
        self._chemberta_helper = {'model': AutoModel.from_pretrained(chemberta_model_name), 'tokenizer': AutoTokenizer.from_pretrained(chemberta_model_name)}
        
        # loading stokes models from github
        self._stokes_chemprop_models = [self._load_model_from_hf_hub(f"checkpoints_model_{i}/best-model-{i}.ckpt") for i in range(1, 21)]

    def _load_model_from_hf_hub(self, filename) -> models.MPNN:
        path = hf_hub_download("jacktnorris/stokes-et-al-chemprop-model", filename)         
        return models.MPNN.load_from_checkpoint(path, map_location=self._tensor_map_location)


    @verify_smiles
    def encode_maccs(self, smiles: str) -> NDArray:
        return np.array(list(Chem.MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))))
    
    @verify_smiles
    def encode_morgan(self, smiles: str, fp_type: Literal["ecfp4", "ecfp6", "fcfp4", "fcfp6"] = "ecfp4") -> NDArray:
        if fp_type not in self._morgan_gens:
            raise ValueError(f"fp_type is not valid: {fp_type}")
        return np.array(list(self._morgan_gens[fp_type].GetFingerprint(Chem.MolFromSmiles(smiles))))
    
    @verify_smiles
    def encode_rdkit(self, smiles: str) -> NDArray:
        return np.array(list(self._rdk_gen.GetFingerprint(Chem.MolFromSmiles(smiles))))
    
    def encode_chemberta(self, smiles: str) -> NDArray:
        # convert smile into token
        inputs = self._chemberta_helper['tokenizer'](smiles, return_tensors="pt", padding=True, truncation=True)
        # using nograd since we're only using inference
        with torch.no_grad():
            outputs = self._chemberta_helper['model'](**inputs)
            hidden_states = outputs.last_hidden_state
            embedding = hidden_states.mean(dim=1).squeeze(0) # mean pooling, collapsing batch dim
            return np.array(embedding)
    @verify_smiles
    def encode_stokes_fingerprint(self, smiles: str) -> NDArray:
        return self._rdk_stokes_featurizer(Chem.MolFromSmiles(smiles))
    
    @verify_smiles
    def encode_stokes_GNN(self, smiles: str) -> NDArray:
        # 1. Featurize the single molecule and prepare a batch of size 1
        mol_data = data.MoleculeDatapoint.from_smi(smiles)
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v1())
        test_dset = data.MoleculeDataset([mol_data], featurizer=featurizer)
        test_loader = data.build_dataloader(test_dset, shuffle=False, batch_size=1)
        
        # 2. Extract the single batch from the loader
        batch = next(iter(test_loader))
        
        all_fingerprints = []
        
        # 3. Iterate through all models and get the fingerprint from each
        # no_grad meaning no gradient calculation
        with torch.no_grad():
            for model in self._stokes_chemprop_models:
                # The model's encoding method takes a batch and an index.
                # We are using index 0 because our batch has only one molecule.
                fingerprint = model.encoding(batch.bmg, batch.V_d, batch.X_d, i=0)
                all_fingerprints.append(fingerprint)
                
        # 4. Concatenate and compute the average
        concatenated_fingerprints = torch.stack(all_fingerprints, dim=0) # shape (num_models, fingerprint_dim)
        average_fingerprint = torch.mean(concatenated_fingerprints, dim=0, keepdim=False)
        
        # 5. Convert the final tensor to a NumPy array for the return type
        return average_fingerprint.numpy().squeeze() # remove batch dim
