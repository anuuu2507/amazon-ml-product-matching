# ML Challenge 2026: Business Entity Resolution Solution

Team Name: Base Case Team Members: K. Sri Anuradha, P. Harshith Kumar Submission Date: 26 September 2026

## 1. Executive Summary

We explored a blocking-and-scoring approach to link businesses across the three supplied datasets using business names, addresses, and country fields. Initial validation showed that broad token overlap can retrieve many true matches but also creates substantial false positives, so the matching rules and full-data pipeline still require further validation.

## 2. Methodology

### 2.1 Problem Analysis

The supplied data contains three large TSV datasets, with Source 1 acting as the reference set and Source 2 and Source 3 containing possible matches.

Training dataset sizes:

- Source 1: 2,206,821 records

- Source 2: 5,034,616 records

- Source 3: 5,285,603 records

Test dataset sizes:

- Source 1: 1,732,544 records

- Source 2: 4,887,273 records

- Source 3: 5,082,316 records

Initial inspection and prediction review showed that business names and addresses can differ between records. Some records have missing addresses, and some examples contain spelling or OCR-like variations. The test set also includes France, which is not present in the training countries inspected, so the approach must not assume only the training countries will occur at test time.

### 2.2 Solution Strategy

The current prototype uses text tokenization and name/address overlap to generate and score possible matches. We created a held-out validation sample and checked whether the known matching IDs could be found in the candidate sources. The full-data matching pipeline and final decision rules are still in development.

Approach Type: Blocking + similarity scoring (prototype)

Core Innovation: Not yet established; the current work is a baseline using separate name and address token overlap.

## 3. Candidate Generation (Blocking)

Blocking keys used: Shared lowercase alphanumeric tokens from business names and addresses; very frequent tokens are filtered in the diagnostic prototype.

Candidate pairs generated: Not yet measured for the final full-data pipeline. The smoke test wrote 100 Source 1 rows and is not a submission result.

How we ensured true matches were not lost: On a diagnostic sample, all 330 distinct true matched IDs were present in the corresponding training Source 2/Source 3 data. A separate token-blocking diagnostic retrieved 309 of those 330 IDs (recall 0.9364) when a fallback was used for missing addresses. This is a small diagnostic sample, not a guarantee of recall on the full dataset.

## 4. Matching Model

Features used:

- Name features: Lowercase alphanumeric tokenization and Jaccard similarity.

- Address features: Lowercase alphanumeric tokenization and Jaccard similarity.

- Other: Country equality is used as a filter when both country values are present. The prototype lowers confidence in name-only comparisons when an address is missing.

Model type: Rule-based similarity scoring prototype; no trained classifier has been finalized.

Threshold selection method: A diagnostic threshold and top-K sweep was run on a small sample. Final threshold selection is pending further validation.

## 5. Results & Error Analysis

F₀.₅ Score (macro): Not yet established for the final system. In one 100-record diagnostic run, the pair-level result at threshold 0.65 was precision 0.1874, recall 0.6424, and F₀.₅ 0.2184. This is not a final or macro leaderboard score.

Common false positives (wrong merges): Similar business names can produce high similarity even when addresses differ. Records with missing addresses can also receive misleadingly high name-only scores.

Common false negatives (missed matches): Name and address spelling differences, OCR-like errors, and low token overlap can cause true matches to fall below the scoring threshold or fail to enter the candidate set.

## 6. Conclusion

The initial experiments established a working data-inspection and validation workflow and demonstrated that candidate generation can recover many true matches on a small sample. They also revealed that the current scoring baseline produces false positives and needs improvement. Full-data matching, final validation, and submission outputs remain to be completed.

## Appendix

### A. Code Artefacts

The project currently contains diagnostic scripts under `src/`, including data inspection, candidate coverage checks, validation scoring, and prediction review. The smoke-test script writes sample output files to `output/`, but those files are not valid full submission outputs. The final runnable entry point and packaged code structure will be documented after the full-data matcher is completed.

### B. Additional Results

- Held-out validation sample created: 22,068 Source 1 records, stratified by the number of known matches.

- A 100-record diagnostic sample contained 330 distinct true matched IDs.

- Exact normalized-name blocking alone retrieved 127 of 330 true IDs (recall 0.3848) in that diagnostic.

- Token blocking with a missing-address fallback retrieved 309 of 330 true IDs (recall 0.9364) in a separate diagnostic.

- These results are exploratory and should not be treated as final submission performance.
