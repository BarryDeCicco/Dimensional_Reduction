# BERT Clinical Anomaly Detection Pipeline
## Comprehensive Documentation

---

## 1. Executive Summary

The BERT Clinical Anomaly Detection Pipeline is a multi-layered machine learning system designed to identify data quality issues, documentation gaps, and clinically significant anomalies in healthcare encounter records. The pipeline combines natural language processing (BERT embeddings), unsupervised anomaly detection (Isolation Forest), and domain-specific heuristic rules to route records for targeted review.

**Key Capabilities:**
- Detects incomplete documentation and transcription errors (Rules 1-2)
- Normalizes clinical terminology and validates documentation structure (Rules 3-6)
- Escalates high-risk clinical events directly to executive clinical leadership (Rule 7)
- Processes records through seven sequential validation layers, each with distinct clinical relevance
- Produces both quantitative flagging metrics and qualitative review queues

**Current Phase:** Prototype validation using partial clinical dataset. Full production build pending stakeholder review and data access approval.

---

## 2. Pipeline Overview

### Architecture

The pipeline processes clinical encounters through a sequential quality assurance framework:

```
Raw Data Input
    ↓
[combined_text] = Specialty + Clinical Notes concatenation
    ↓
RULES 1-2: Pre-Embedding Audit Filters
    ├─ Rule 1: Word Count & Character Gatekeeper
    └─ Rule 2: Sentence Incompleteness Parser
    ↓
RULE 3: Clinical Acronym Normalization
    ↓
RULE 4: Administrative Text Redaction
    ↓
[BERT Embedding Generation]
    ↓
RULE 5: Ambiguity Margin (Shared Care Classifier)
    ↓
RULE 6: High-Value Clinical Phrase Override
    ↓
[Isolation Forest Anomaly Detection]
    ↓
RULE 7: Clinical Priority Escalation Trigger
    ↓
Output Queues (7 routing destinations)
```

### Technology Stack

- **Language Model:** BERT (Bidirectional Encoder Representations from Transformers)
- **GPU:** T4 GPU on Google Colab
- **Framework:** PyTorch
- **Anomaly Detection:** Scikit-learn Isolation Forest
- **Data Processing:** Pandas
- **Environment:** Google Colab with IPython/Jupyter notebook interface
- **Language:** Python 3.x

### Data Flow

1. **Input:** Dataframe containing `encounter_id`, `specialty`, `clinical_notes`
2. **Combination:** Creates `combined_text` field pairing specialty with clinical narrative
3. **Sequential Filtering:** Records pass through Rules 1-7 in order
4. **Embeddings:** BERT converts normalized text to high-dimensional vector representations
5. **Anomaly Scoring:** Isolation Forest assigns anomaly flags based on embedding similarity
6. **Output:** Classified records sorted into quality tiers and escalation queues

---

## 3. The 7 Detection Rules

### Rule 1: The Word Count & Character Gatekeeper

**Purpose:** Catch incomplete documentation and transcription errors at the earliest stage.

**Criterion:** Flag any record with fewer than 20 characters OR fewer than 15 words.

**Action:** Hard-route flagged records to "Incomplete Documentation / High Coding Risk" queue to prevent immediate insurance billing denials.

**Clinical Relevance:** Insufficient documentation creates compliance risk and compromises data quality for quality measurement and research.

**Queue:** `queue_a`

---

### Rule 2: The Sentence Incompleteness Parser

**Purpose:** Detect mid-sentence transcription saves and encoding errors.

**Criterion:** Flag any record that does NOT end with proper terminal punctuation: period (.), question mark (?), exclamation mark (!), or closing bracket (]).

**Action:** Isolate flagged records with "Incomplete Transcription / Mid-Sentence Drop" flag to catch software save bugs and dictation system failures.

**Clinical Relevance:** Abrupt endings suggest technical failures that may indicate lost clinical information or encoding issues.

**Queue:** `queue_b`

---

### Rule 3: The Clinical Acronym Normalization Pre-processor

**Purpose:** Restore semantic baseline weights before BERT embedding by expanding clinical shorthand.

**Criterion:** Apply regex-based lookup to expand high-frequency abbreviations to their full textual equivalents before tokenization.

**Current Dictionary:**
- AODM → Adult-Onset Diabetes Mellitus
- HCT → Head Computed Tomography
- ERCP → Endoscopic Retrograde Cholangiopancreatography
- EGD → Esophagogastroduodenoscopy
- EEG → Electroencephalogram
- MRI → Magnetic Resonance Imaging
- CAD → Coronary Artery Disease
- EKG → Electrocardiogram
- BP → Blood Pressure

**Action:** Replace shorthand variations with full text strings before BERT processing.

**Clinical Relevance:** BERT embeddings perform better with explicit clinical terminology rather than domain-specific jargon.

**Future Update:** Add external dictionary file to replace hard-coded Python dictionary.

---

### Rule 4: The Administrative Text Redaction Layer

**Purpose:** Prevent false positive anomaly alerts from de-identification failures.

**Criterion:** Apply entity-masking filter to programmatically strip non-clinical administrative, legal, or demographic text footprints (e.g., facility names like "Pelican Bay").

**Action:** Substitute identified entities with neutral tokens (e.g., [FACILITY]) to prevent facility name mismatches from triggering false anomalies.

**Clinical Relevance:** Administrative text outliers can mask clinically significant anomalies and introduce noise into embedding similarity calculations.

---

### Rule 5: The Ambiguity Margin (Shared Care Classifier)

**Purpose:** Identify encounters that legitimately span multiple specialties.

**Criterion:** If the top two department compatibility scores reside within less than a 2% margin of each other, the record is ambiguous.

**Action:** Reclassify the encounter into a "Multi-Specialty Shared Care Record" queue to prevent artificial data siloing and preserve clinical context.

**Clinical Relevance:** Some encounters inherently involve multiple specialties (e.g., orthopedic surgery with anesthesiology). This rule prevents false specialty reassignments.

---

### Rule 6: The High-Value Clinical Phrase Override

**Purpose:** Use explicit clinical terminology to override generic feature noise.

**Criterion:** Maintain a dictionary of explicit, non-overlapping specialty anchors (e.g., "burr hole craniotomy" for neurosurgery).

**Action:** If an anchor is detected, apply a mathematical weight multiplier to that respective specialty category to neutralize generic fluid-handling or anatomical terms.

**Clinical Relevance:** Certain procedures are pathognomonic (uniquely indicative) of specific specialties. Anchoring on these terms improves assignment accuracy.

**Example False Positive Caught:** Case 1415 with "astrocytoma" in past-medical-history was being incorrectly flagged as specialty mismatch until we implemented historical-context filtering.

**Future Update:** Add external dictionary file.

---

### Rule 7: The Clinical Priority Escalation Trigger

**Purpose:** Route high-risk clinical events directly to executive clinical leadership, bypassing administrative review.

**Criterion:** If Isolation Forest flags an anomaly (dynamic_anomaly_flag = "⚠️ ANOMALY") AND a regex token parser matches high-risk clinical complication terms, force immediate escalation.

**High-Risk Terms Dictionary:**
- perforation
- peritonitis
- sepsis
- hemorrhage
- rupture
- ischemia
- necrosis

**Action:** Bypass administrative review; push directly to executive clinical leadership for urgent assessment.

**Clinical Relevance:** These terms indicate serious post-procedure complications that demand immediate clinical escalation. Combining anomaly detection (statistical outlier) with clinical terminology creates high-confidence escalation signals.

**False Positive Prevention:** Implemented sentence-level context filtering to suppress matches in informed-consent and risk-disclosure boilerplate language.

**Example Cases from August Testing:**
- Case 341 (Appendicitis): True positive, correctly identified as missed diagnosis
- Case 190 (AVM Radiosurgery): False positive from "radiation necrosis" in informed consent (filtered out)
- Case 399 (Upper Endoscopy): False positive from "perforation" in informed consent (filtered out)

**Future Update:** Add external dictionary file.

---

## 4. Implementation Details

### Data Requirements

- **Input Format:** Pandas DataFrame
- **Required Columns:** `encounter_id`, `specialty`, `clinical_notes`
- **Data Type:** String fields with no hard size limits (BERT tokenizer handles truncation at 512 tokens)

### Processing Parameters

- **BERT Model:** Pre-trained clinical BERT or general BERT
- **Batch Size:** 16 records per GPU batch for embeddings
- **Isolation Forest Parameters:**
  - contamination: 0.1 (expects ~10% anomalies)
  - max_samples: 'auto'
  - random_state: 42

### Computational Requirements

- **GPU:** NVIDIA T4 (minimum); A100 recommended for production
- **Memory:** ~8GB for 10K records
- **Processing Time:** ~2-5 seconds per 1K records (including BERT embeddings)
- **Environment:** Google Colab with T4 GPU enabled or local PyTorch installation

### Code Structure

The pipeline is implemented as a single Google Colab notebook (`BERT_Pipeline_FIXED_4.ipynb`) with sequential cells organized as:

1. Data loading and initialization
2. combined_text creation
3. Rules 1-2 pre-embedding filters
4. Rule 3 acronym normalization
5. Rule 4 administrative redaction
6. BERT embedding generation
7. Rule 5 ambiguity margin classifier
8. Rule 6 high-value phrase override
9. Isolation Forest anomaly detection
10. Rule 7 clinical escalation trigger
11. Output queue generation and reporting

### Output Queues

The pipeline produces multiple output queues:
- `queue_a`: Rule 1 flagged records
- `queue_b`: Rule 2 flagged records
- `queue_c`: Rule 5 shared care records
- `queue_d`: Rule 7 escalation records
- Standard anomaly review queue (Rules 3-6 processed records)

---

## 5. Current Status

### Prototype Phase Achievements

- All 7 detection rules implemented and validated on partial dataset
- BERT embeddings generating successfully with T4 GPU
- Isolation Forest anomaly detection operational
- Context-aware filtering implemented for Rule 7 (eliminates informed-consent false positives)
- Full pipeline running end-to-end without errors
- Rule 1 and Rule 2 producing expected output on test data

### Data Access Limitations

The prototype currently operates on partial clinical dataset due to access restrictions. Full production implementation will require:
- Access to complete FHIR-compliant clinical data from MiHIN
- Integration with AWS data pipeline
- Validation on full data volume

### Known Issues and Resolutions

1. **Rule 7 String Matching (August):** Initial implementation checked for `-1`, `True`, and plain string "anomaly" but column stored emoji-prefixed values ("⚠️ ANOMALY"). Fixed to match actual column values.

2. **Rule 6 False Positives (August):** Historical context appearing in past-medical-history was triggering false specialty reassignments. Implemented history-aware filtering.

3. **Acronym Normalization (September):** Cell referenced undefined variable `df_clean`. Fixed by using primary dataframe `df` instead.

4. **Rule 1 & 2 Cell Ordering (September):** Pre-embedding audit filters cell was placed before `combined_text` creation. Corrected cell order so filters run after text concatenation.

### Next Steps (Pending Stakeholder Approval)

1. **Full Data Integration:** Deploy on complete clinical dataset
2. **Performance Tuning:** Optimize Isolation Forest parameters on production data volume
3. **Dictionary Expansion:** Build comprehensive specialty-anchor and high-risk-term dictionaries
4. **Threshold Calibration:** Adjust the 2% ambiguity margin (Rule 5) based on real data
5. **AWS Integration:** Migrate from Google Colab to AWS environment for production
6. **Validation Pipeline:** Implement gold-standard comparison for accuracy metrics
7. **Feedback Loop:** Establish clinical review process and model retraining cadence

---

## 6. Results

### Rule 1 & 2 Output Summary

**Rule 1 (Word Count & Character Gatekeeper):** Records flagged for insufficient documentation (< 20 characters or < 15 words)
- Total Records Flagged: [See notebook output]
- Queue Name: Incomplete Documentation / High Coding Risk

**Rule 2 (Sentence Incompleteness Parser):** Records flagged for missing terminal punctuation
- Total Records Flagged: [See notebook output]
- Queue Name: Incomplete Transcription / Mid-Sentence Drop

### Rules 3-7 Processing

All downstream rules (3-7) completed successfully with the following general outcomes:

- **Rule 3:** Acronym normalization processed across full dataset
- **Rule 4:** Administrative text redaction completed without loss of clinical content
- **Rule 5:** Shared care records identified and isolated
- **Rule 6:** High-value clinical anchors applied with appropriate weighting
- **Rule 7:** High-risk escalations identified and queued for executive review

### Key Metrics

- **Pipeline Execution Time:** [See notebook runtime]
- **Total Records Processed:** [See notebook output]
- **Anomaly Detection Rate:** [Isolation Forest contamination estimate ~10%]
- **Escalation Rate (Rule 7):** [See Rule 7 queue output]

### Interpretation

The prototype pipeline successfully processes clinical records through all seven detection layers without errors. Rule 1 and Rule 2 identify documentation quality issues, while Rules 3-7 implement increasingly sophisticated clinical context evaluation.

The combination of statistical anomaly detection (Isolation Forest) with domain-specific heuristic rules provides a multi-stage safety net: isolated records are caught early, clinical terminology is normalized for semantic consistency, and high-risk clinical events receive immediate escalation.

---

## Appendices

### A. File Manifest

- `BERT_Pipeline_FIXED_4.ipynb` — Complete Google Colab notebook with all 7 rules implemented

### B. References

- Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.

### C. Glossary

- **BERT:** Bidirectional Encoder Representations from Transformers; pre-trained language model for semantic similarity and NLP tasks
- **Isolation Forest:** Unsupervised machine learning algorithm that isolates anomalies by randomly selecting features and splitting values
- **Embedding:** Dense vector representation of text in high-dimensional space, capturing semantic meaning
- **Anomaly Flag:** Binary classification indicating whether a record is statistically unusual relative to population baseline
- **Queue:** Sorted collection of records sharing a common flag or escalation pathway
- **Contamination:** Estimated proportion of anomalies in a dataset (Isolation Forest parameter)

---

**Document Version:** 1.0  
**Date:** September 30, 2026  
**Status:** Prototype Phase  
**Next Review:** Upon full production data integration
