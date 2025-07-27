"""
Machine learning workflow example.

This example demonstrates how to:
1. Generate molecular datasets
2. Extract features
3. Train ML models
4. Make predictions with uncertainty
"""

import numpy as np

from quantum.core import Molecule
from quantum.ml import FeatureExtractor, MolecularML


def generate_sample_molecules(n_molecules=100):
    """Generate sample molecules for demonstration."""
    molecules = []
    properties = []

    # Create simple hydrocarbon molecules
    np.random.seed(42)

    for i in range(n_molecules):
        # Random number of carbons (1-6)
        n_carbons = np.random.randint(1, 7)

        # Create linear alkane
        atoms = ["C"] * n_carbons

        # Add hydrogens (simplified)
        n_hydrogens = 2 * n_carbons + 2
        atoms.extend(["H"] * n_hydrogens)

        # Generate random coordinates
        coords = np.random.random((len(atoms), 3)) * 5

        # Create molecule
        mol = Molecule(
            name=f"alkane_C{n_carbons}H{n_hydrogens}",
            atoms=atoms,
            coordinates=coords,
        )

        molecules.append(mol)

        # Simulate property (energy) with some correlation to molecular size
        energy = -n_carbons * 40.0 + np.random.normal(0, 2.0)
        properties.append(energy)

    return molecules, properties


def demonstrate_feature_extraction():
    """Demonstrate different feature extraction methods."""
    print("Feature Extraction Demonstration")
    print("=" * 40)

    # Create a simple molecule
    mol = Molecule(
        name="methane",
        atoms=["C", "H", "H", "H", "H"],
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ),
    )

    # Test different feature extraction methods
    methods = [
        "coulomb_matrix",
        "bag_of_bonds",
        "descriptors_3d",
        "morgan_fingerprints",
    ]

    for method in methods:
        try:
            extractor = FeatureExtractor(method=method)
            features = extractor.extract_molecular_features(mol)
            feature_names = extractor.get_feature_names()

            print(f"\n{method.upper()}:")
            print(f"  Number of features: {len(features)}")
            print(f"  Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
            print(f"  First 5 features: {features[:5]}")

        except Exception as e:
            print(f"\n{method.upper()}: Failed - {e}")


def main():
    """Run machine learning workflow example."""
    print("QuantChem Machine Learning Workflow Example")
    print("=" * 50)

    # Demonstrate feature extraction
    demonstrate_feature_extraction()
    print()

    # Generate training data
    print("Generating sample dataset...")
    molecules, energies = generate_sample_molecules(200)
    print(f"Generated {len(molecules)} molecules")
    print(f"Energy range: [{np.min(energies):.2f}, {np.max(energies):.2f}] Hartree")
    print()

    # Test different ML models and features
    model_configs = [
        ("random_forest", "coulomb_matrix"),
        ("random_forest", "descriptors_3d"),
        ("neural_network", "coulomb_matrix"),
    ]

    for model_type, feature_type in model_configs:
        print(f"Training {model_type} with {feature_type} features...")

        try:
            # Initialize ML model
            ml_model = MolecularML(
                model_type=model_type,
                feature_type=feature_type,
                random_state=42,
            )

            # Train model
            results = ml_model.train(molecules, energies, test_size=0.2)

            print("  Training results:")
            print(f"    MAE: {results.mae:.4f} Hartree")
            print(f"    RMSE: {results.rmse:.4f} Hartree")
            print(f"    R²: {results.r2:.4f}")
            print(f"    Training size: {results.training_size}")
            print(f"    Test size: {results.test_size}")

            # Feature importance (if available)
            importance = ml_model.feature_importance()
            if importance:
                top_features = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )[:5]
                print("    Top 5 features:")
                for feature, score in top_features:
                    print(f"      {feature}: {score:.4f}")

            # Make predictions on new molecules
            test_molecules = molecules[:5]  # Use first 5 as test
            predictions = ml_model.predict(test_molecules)
            actual_values = energies[:5]

            print("  Sample predictions:")
            for i, (pred, actual) in enumerate(zip(predictions, actual_values)):
                error = abs(pred - actual)
                print(
                    f"    Molecule {i+1}: Pred={pred:.3f}, Actual={actual:.3f}, Error={error:.3f}"
                )

            # Uncertainty estimation
            if model_type == "random_forest":
                pred_mean, pred_std = ml_model.predict_with_uncertainty(
                    test_molecules[:3]
                )
                print("  Predictions with uncertainty:")
                for i, (mean, std) in enumerate(zip(pred_mean, pred_std)):
                    print(f"    Molecule {i+1}: {mean:.3f} ± {std:.3f}")

            print()

        except Exception as e:
            print(f"  Training failed: {e}")
            print()

    # Cross-validation example
    print("Performing cross-validation...")
    try:
        ml_model = MolecularML(
            model_type="random_forest",
            feature_type="coulomb_matrix",
            random_state=42,
        )

        cv_results = ml_model.cross_validate(molecules, energies, cv_folds=5)

        print("  Cross-validation results (5-fold):")
        print(f"    MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
        print(f"    MSE: {cv_results['mse_mean']:.4f} ± {cv_results['mse_std']:.4f}")
        print(f"    R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")

    except Exception as e:
        print(f"  Cross-validation failed: {e}")

    print()

    # Model saving and loading example
    print("Model persistence example...")
    try:
        # Train a model
        ml_model = MolecularML(
            model_type="random_forest",
            feature_type="coulomb_matrix",
            random_state=42,
        )

        ml_model.train(molecules, energies, test_size=0.2)

        # Save model
        model_path = "example_model.joblib"
        ml_model.save_model(model_path)
        print(f"  Model saved to {model_path}")

        # Load model and make prediction
        new_ml_model = MolecularML()
        new_ml_model.load_model(model_path)

        prediction = new_ml_model.predict(molecules[0])
        print(f"  Loaded model prediction: {prediction:.4f}")

        # Clean up
        import os

        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"  Cleaned up {model_path}")

    except Exception as e:
        print(f"  Model persistence failed: {e}")


if __name__ == "__main__":
    main()
