"""
Healthcare Data Preprocessing Pipeline 
Dynamically extracts symptoms & precautions without ANY hardcoded mappings.
"""

import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict


class HealthcareDataPreprocessor:

    def __init__(self):
        pass

    # ============================================================
    #            DISEASE NORMALIZATION (GENERIC ONLY)
    # ============================================================
    def normalize_disease_name(self, name: str) -> str:
        """Normalize disease names WITHOUT static corrections."""

        if pd.isna(name):
            return ""

        name = str(name).strip().lower()

        # Remove parentheses but keep enclosed words
        name = re.sub(r"[()]", "", name)

        # Normalize whitespace
        name = re.sub(r"\s+", " ", name)

        return name.strip()

    # ============================================================
    #                    CLEAN SYMPTOM TEXT
    # ============================================================
    def clean_symptom_text(self, text: str) -> str:
        """Normalize symptom text generically."""

        if pd.isna(text):
            return ""

        text = str(text).strip().lower()

        # Replace underscores with spaces
        text = text.replace("_", " ")

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ============================================================
    #      AUTOMATIC ROOT-BASED SYMPTOM SYNONYM GENERATION
    # ============================================================
    def generate_dynamic_synonyms(self, symptoms: list) -> dict:
        """
        Generates lightweight "root-based" synonym clusters dynamically.
        No static dictionary is used.
        """

        root_map = defaultdict(set)

        for s in symptoms:
            root = s
            root = re.sub(r"ing$", "", root)   # itching → itch
            root = re.sub(r"s$", "", root)     # pains → pain
            root_map[root].add(s)

        return {root: list(values) for root, values in root_map.items()}

    # ============================================================
    #            PROCESS SYMPTOM DATASET (DYNAMIC)
    # ============================================================
    def process_symptom_dataset(self, filepath: str) -> pd.DataFrame:
        print(f"\nProcessing Symptom Dataset → {filepath}")

        df = pd.read_csv(filepath)
        print(f"  Loaded: {df.shape} rows")

        df["clean_disease"] = df["Disease"].apply(self.normalize_disease_name)

        # Auto-detect symptom columns
        symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
        print(f"  Found {len(symptom_cols)} symptom columns")

        grouped = defaultdict(list)

        for _, row in df.iterrows():
            disease = row["clean_disease"]

            for col in symptom_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    cleaned = self.clean_symptom_text(value)
                    if cleaned:
                        grouped[disease].append(cleaned)

        processed_data = []

        for disease, symptoms in grouped.items():

            symptoms = list(set(symptoms))  # remove duplicates

            # Generate dynamic synonym expansions
            synonym_map = self.generate_dynamic_synonyms(symptoms)

            expanded = set(symptoms)
            for root, words in synonym_map.items():
                expanded.update(words)

            processed_data.append({
                "disease": disease,
                "symptoms": symptoms,
                "symptoms_text": " ".join(symptoms),
                "expanded_symptoms": list(expanded),
                "expanded_text": " ".join(expanded),
                "symptoms_count": len(symptoms)
            })

        processed_df = pd.DataFrame(processed_data)
        print(f"  Processed diseases: {processed_df.shape[0]}")

        return processed_df

    # ============================================================
    #              PROCESS PRECAUTION DATASET
    # ============================================================
    def process_precaution_dataset(self, filepath: str) -> pd.DataFrame:
        print(f"\nProcessing Precaution Dataset → {filepath}")

        df = pd.read_csv(filepath)
        print(f"  Loaded: {df.shape} rows")

        df["clean_disease"] = df["Disease"].apply(self.normalize_disease_name)

        precaution_cols = [c for c in df.columns if c.lower().startswith("precaution")]

        processed = []

        for _, row in df.iterrows():
            disease = row["clean_disease"]

            precautions = []
            for col in precaution_cols:
                p = row[col]
                if pd.notna(p) and str(p).strip():
                    p = str(p).strip().lower()
                    p = re.sub(r"\s+", " ", p)
                    precautions.append(p)

            processed.append({
                "disease": disease,
                "precautions": precautions,
                "precautions_text": " ".join(precautions),
                "precautions_count": len(precautions)
            })

        return pd.DataFrame(processed)

    # ============================================================
    #              CREATE RETRIEVAL DATASET
    # ============================================================
    def create_retrieval_dataset(self, symptom_df, precaution_df):
        print("\nMerging Symptom + Precaution data...")

        merged = symptom_df.merge(
            precaution_df,
            on="disease",
            how="left",
            suffixes=("", "_prec")
        )

        merged["precautions"] = merged["precautions"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        merged["precautions_text"] = merged["precautions_text"].fillna("")
        merged["precautions_count"] = merged["precautions_count"].fillna(0)

        merged["search_text"] = merged.apply(
            lambda r: f"{r['disease']} {r['expanded_text']} {r['precautions_text']}".strip(),
            axis=1
        )

        def display(row):
            text = f"Disease: {row['disease'].title()}\n\nSymptoms:\n"
            for i, s in enumerate(row["symptoms"], 1):
                text += f"  {i}. {s}\n"

            if row["precautions"]:
                text += "\nPrecautions:\n"
                for i, p in enumerate(row["precautions"], 1):
                    text += f"  {i}. {p}\n"

            return text

        merged["display_text"] = merged.apply(display, axis=1)
        merged["doc_id"] = range(len(merged))

        print(f"  Final dataset shape: {merged.shape}")

        return merged

    # ============================================================
    #                SAVE OUTPUT FILES
    # ============================================================
    def save_processed_data(self, df, output_dir="data/processed"):
        os.makedirs(output_dir, exist_ok=True)

        out_path = os.path.join(output_dir, "retrieval_dataset.csv")
        df.to_csv(out_path, index=False)

        print(f"\n✅ Saved retrieval dataset → {out_path}")

        return out_path

    # ============================================================
    #                RUN PIPELINE
    # ============================================================
    def run_pipeline(self, symptom_file, precaution_file):
        print("=" * 60)
        print("     HEALTHCARE DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        s_df = self.process_symptom_dataset(symptom_file)
        p_df = self.process_precaution_dataset(precaution_file)

        retrieval_df = self.create_retrieval_dataset(s_df, p_df)

        self.save_processed_data(retrieval_df)

        print("\nPipeline completed successfully!")
        print(f"Documents: {len(retrieval_df)}")
        print(f"Unique diseases: {retrieval_df['disease'].nunique()}")

        return retrieval_df


def main():
    pre = HealthcareDataPreprocessor()
    pre.run_pipeline("data/raw/DiseaseAndSymptoms.csv",
                     "data/raw/DiseasePrecautions.csv")


if __name__ == "__main__":
    main()
