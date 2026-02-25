import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os

def compute_ecfp4(smiles_list, radius=2, nbits=2048):
    """
    Convert SMILES strings into ECFP4 fingerprints.
    
    What is a fingerprint?
    - Each molecule becomes a list of 2048 numbers (0 or 1)
    - Each number represents whether a certain chemical 
      pattern is present in the molecule
    - This is how we turn chemistry into numbers for ML
    
    radius=2 means we look 2 bonds away from each atom
    nbits=2048 means we use 2048 slots to store patterns
    """
    fingerprints = []
    failed = 0

    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Use zeros if molecule fails
                fingerprints.append(np.zeros(nbits))
                failed += 1
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=radius, nBits=nbits
                )
                fingerprints.append(np.array(fp))
        except:
            fingerprints.append(np.zeros(nbits))
            failed += 1

    if failed > 0:
        print(f"  Warning: {failed} molecules failed fingerprinting")

    return np.array(fingerprints)


def process_dataset(csv_path, dataset_name, output_dir):
    print(f"\n{'='*60}")
    print(f"Computing fingerprints for: {dataset_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"Molecules: {len(df)}")
    print(f"Computing ECFP4 fingerprints (2048 bits)...")

    # Compute fingerprints
    fps = compute_ecfp4(df['smiles'].tolist())

    print(f"Fingerprint matrix shape: {fps.shape}")
    print(f"  = {fps.shape[0]} molecules x {fps.shape[1]} features")

    # Save fingerprints
    os.makedirs(output_dir, exist_ok=True)
    fp_path = os.path.join(output_dir, f"{dataset_name}_ecfp4.npy")
    np.save(fp_path, fps)
    print(f"Saved to: {fp_path}")

    # Also save the SMILES order so we know which row = which molecule
    order_path = os.path.join(output_dir, f"{dataset_name}_smiles_order.csv")
    df[['smiles']].to_csv(order_path, index=True)
    print(f"SMILES order saved to: {order_path}")

    return fps


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    # Process Tox21
    fps_tox21 = process_dataset(
        csv_path     = "data/processed/tox21_clean.csv",
        dataset_name = "tox21",
        output_dir   = "data/processed"
    )

    # Process ClinTox
    fps_clintox = process_dataset(
        csv_path     = "data/processed/clintox_clean.csv",
        dataset_name = "clintox",
        output_dir   = "data/processed"
    )

    print(f"\n{'='*60}")
    print(f"ALL FINGERPRINTS COMPUTED")
    print(f"Tox21:   {fps_tox21.shape}")
    print(f"ClinTox: {fps_clintox.shape}")
    print(f"{'='*60}")
