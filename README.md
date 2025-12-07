# Healthcare Information Retrieval: Symptom-to-Disease Retrieval with Evidence-based Recommendations
=========================================================================
Boolean • TF-IDF • BM25 • Hybrid Fusion • Evidence Retrieval • Evaluation
-------------------------------------------------------------------------
This project implements a complete Healthcare Information Retrieval (IR) System that predicts the most likely disease from user-given symptoms by combining multiple retrieval techniques including Boolean search, TF-IDF, BM25, and a Hybrid Fusion model. The system begins by preprocessing medical datasets containing symptoms and precautions, transforming them into a structured retrieval format. It then builds three retrieval engines—Boolean, TF-IDF, and BM25—to score diseases based on symptom similarity. A separate Evidence Retriever uses a curated PubMed-style dataset to fetch supporting medical research for each predicted disease. The central module integrates all models to provide ranked diseases, matched symptoms, precautions, and scientific evidence. Finally, an evaluation module computes IR metrics such as Precision@K, Recall@K, nDCG, and MAP to assess accuracy. The project demonstrates a full end-to-end IR pipeline for healthcare decision support—from data preprocessing to retrieval, fusion, evidence linking, and evaluation.

Given symptoms → return most probable diseases + precautions + medical evidence + evaluation metrics.

This README explains how to run the entire project on your system:
✔ Environment setup
✔ Required libraries
✔ Folder structure
✔ What files to run
✔ Commands for PowerShell / CMD
✔ Expected outputs


########################################################################
Demo Video URL:

    - https://www.loom.com/share/4378d72f98e6481083d5c8de78e3eddc

########################################################################
**********************
1. PROJECT STRUCTURE
**********************
HealthCare_IR_System/
│
├── data/
│   ├── raw/
│   │     └── archive.zip
│   │     └── DiseaseAndSymptoms.csv
│   │     └── DiseasePrecautions.csv
│   ├── processed/
│   │     └── retrieval_dataset.csv (auto-created by preprocessing)
│   ├── indices/
│   │     └── tfidf_index.json (auto-created)
│   └── evidence/
│         └── pubmed_clean.csv
│
├── src/
│   ├── preprocessing/
│   │     └── data_preprocessor.py
│   ├── retrieval/
│   │     ├── boolean_retriever.py
│   │     ├── tfidf_retriever.py
│   │     ├── bm25_retriever.py
│   ├── evidence/
│   │     └── evidence_retriever.py
│   ├── evaluation/
│   │     ├── evaluation.py
│   │     └── evaluation_runner.py
│   └── complete_healthcare_ir_system.py
│
└── README.md

##########################################################################
******************************
2. PYTHON & ENVIRONMENT SETUP
******************************
    Step 1 —  Install Python 3.10+
            - After Intalled Python into system verify it:
        PowerShell / CMD:
            - python --version 
            OR
            - py --version

    Step 2 — Create Virtual Environment
        Create environment:
            Windows PowerShell / CMD: 
                - python -m venv venv
                OR 
                - py -m venv venv (Means install/ created already)
        Activate the environment:
            PowerShell
                - venv\Scripts\Activate (is now activate)

#########################################################################
****************************
3. INSTALL REQUIRED PACKAGES
****************************
    Inside the activated environment:
       - pip install pandas numpy nltk scikit-learn
       - pip install python-dateutil
       - pip install textwrap3
       OR 
    Outside the environment:
       - .\venv\Scripts\python.exe -m pip install pandas numpy nltk scikit-learn python-dateutil textwrap3
        (Package are installed if not install before)
##########################################################################
*****************************
4. PREPARE DATA FILES
*****************************
    - Download Kaggle dataset and add into the directory 
        - data\raw\DiseaseAndSymptoms.csv
        - data\raw\DiseasePrecautions.csv
    - Make Evidence against Diseases
        - data\evidence\pubmed_clean.csv

##########################################################################
******************************
5. Run Notebooks - data exploration
******************************
    Run On PowerShell:
        - py notebooks/01_data_exploration.py

    Output:
        1. Loaded two raw datasets
            DiseaseAndSymptoms.csv 
            DiseasePrecautions.csv 
        2. Checked dataset structure
            Verified column names, total rows, and data types
        3. Handled missing values
            Many symptom columns had high missing percentages 
            Precaution dataset had almost no missing values
            Decision: extract only non-empty symptoms for each disease
        4. Computed symptom statistics
        5. Computed precaution statistics
        6. Verified dataset consistency
        7. Explored sample rows
        8. Generated exploration report
        9. Saved to: data/processed/data_analysis_report.txt
            Contains summary stats, missing values, unique term counts, and dataset comparison

############################################################################
*****************************************
6. Preprocessing Pipeline
*****************************************
    Run On PowerShell:
        - py src/preprocessing/data_preprocessor.py
    
    Output:
        1. Loaded the raw symptom dataset
            - File: data/raw/DiseaseAndSymptoms.csv
            - Shape: 4920 rows × 18 columns
            - Identified 17 symptom columns
        2. Extracted unique diseases
            - Dataset contains multiple rows per disease
            - After processing → 41 unique diseases
        3. Cleaned & normalized symptoms
            - Removed missing & empty symptom values
            - Converted symptom columns into a single combined list
            - Normalized text (lowercase, removed underscores, stripped spaces)
        4. Loaded the raw precaution dataset
            - File: data/raw/DiseasePrecautions.csv
            - Shape: 41 rows × 5 columns
            - Each disease contains 3–4 precaution steps
        5. Merged symptoms + precautions
            - Joined both datasets by disease name
            - Created unified structure for retrieval
            - Final processed dataset shape: 41 rows × 12 columns
        6. Generated expanded symptoms list
            - Includes individual symptom tokens
            - Improves TF-IDF & BM25 retrieval performance
        7. Saved final dataset for retrieval models
            - Output 
                file: data/processed/retrieval_dataset.csv
                Contains:
                    Disease
                    Symptoms (list)
                    Expanded symptoms
                    Precautions (list)
        8. Pipeline completed successfully
            - Total documents prepared: 41
            - All diseases matched correctly with precautions

############################################################################
*****************************************
7. Boolean Retrieval System
*****************************************
    Powershell:
        - py src/retrieval/boolean_retriever.py
    1. Purpose
        - Implements a simple, fast retrieval model for symptom-based disease search.
        - Uses Boolean logic (AND / OR) to match user symptoms against indexed medical data.
    2. Dataset Loading
        - Loads retrieval_dataset.csv from data/processed/.
        - Parses the following list-based columns:
            - symptoms
            - expanded_symptoms
            - precautions
        - Handles CSV list strings safely using ast.literal_eval.
        - Prints:
            - total documents
            - unique diseases
            - sample diseases
    3. Builds an inverted index mapping:
             term → set of document IDs.
    4. Index includes:
            - Full symptom phrases
            - Tokenized individual words (length > 2)
    5. Returns detailed result objects (disease, symptoms, precautions, score).
    6. Formats clean terminal output for readability.
    7. Terminal Testing
        It performs:
            - Loading dataset
            - Building index
            - Printing sample index
            - Running predefined queries
            - Starting interactive search mode
############################################################################
*****************************************
8. TF-IDF Retriever
*****************************************
    Powershell:
        - py src/retrieval/tfidf_retriever.py
    1. Loads retrieval_dataset.csv
    2. Builds TF-IDF vectors for all 41 diseases
    3. Saves/loads index file:
        - data/indices/tfidf_index.json
    4. Tokenizes query into TF-IDF vector
    5. Computes cosine similarity
    6. Returns top-ranked diseases
    7. Supports interactive query mode

############################################################################
*****************************************
9. BM25 Retriever
*****************************************
    Command:
        - py src/retrieval/bm25_retriever.py   
    1. Tokenizes expanded symptoms
    2. Computes BM25 scores using:  
        k1 = 1.5
        b = 0.75
    3.  Ranks diseases by probability
            Displays:
                Disease
                Score
                Matched terms
                Precaution
    5. Interactive testing mode included

############################################################################
*****************************************
10. Evidence Retriever
*****************************************
    Command:
        - py src/evidence/evidence_retriever.py
    Functions performed:
    1. Loads evidence file (75 articles)
    2. Tokenizes title + abstract
    3. Builds BM25 index
    4. Matches evidence relevant to disease
    Outputs:
        Article title
        BM25 score
        Summary
        First sentences from abstract

############################################################################
*****************************************
12. HYBRID FUSION (Boolean + TF-IDF + BM25)
*****************************************
    Command:
        - py src/complete_healthcare_ir_system.py
    Steps performed in hybrid search:
        1. Normalizes each model score
        2. Combines using weighted fusion:
            Boolean: 0.2
            TF-IDF: 0.3
            BM25: 0.5
        3. Returns final disease ranking
        4. Adds precautions
        5. Adds evidence support
        6. Returns clean combined output

############################################################################
*****************************************
13. COMPLETE HEALTHCARE IR SYSTEM
*****************************************
    Command:
        - py src/complete_healthcare_ir_system.py
    System loads:
        - Boolean index
        - TF-IDF index
        - BM25 index
        - Evidence retriever
    Testing Terms Commands:
        skin rash
        boolean skin rash
        tfidf skin rash
        bm25 skin rash
        compare skin rash
        stats
        quit
    Outputs:
        - Disease prediction
        - Matched symptoms
        - Combined score
        - Precautions
        - Research evidence
    

############################################################################
*****************************************
14. EVALUATION MODULE
*****************************************
    Command:
        - py src/evaluation/evaluation_runner.py

    Provides IR metrics:
        - Precision@K
        - Recall@K
        - F1@K
        - nDCG@K
        - Average Precision (AP)
        - Mean Average Precision (MAP)

    Process:
        - Define ground-truth document IDs for each query
        - System retrieves results
        - Metrics are auto-calculated
        - Final MAP displayed for entire model


########################################################################
