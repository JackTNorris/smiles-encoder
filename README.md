Very basic shared utility class for encoding chemical compound strings (SMILES) into vector representations

To import (para importar):

- Run `pip install git+https://github.com/jacktnorris/smiles-encoder.git`
- Import in your python script with `from smiles_encoder import SMILESEncoder`
- Note: ensure that your python version is at least 3.13.

Other notes:
- By default, the Chemprop models and Chemberta models are not loaded (this is for performance purposes)
- If you wish to enable the use of chemprop and chemberta, set the `load_models` flag in your constructor to `True`. The class will take longer to load, and your constructor will take a second to be called, but it should execute faster after the first import due to model caching
