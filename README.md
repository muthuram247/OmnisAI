
# Omnis AI

**Evidence-first document question-answering assistant**

Omnis AI is a lightweight document retrieval application that helps users find relevant information in TXT and Markdown files.

## Features

- Upload TXT and Markdown documents
- Split documents into searchable text chunks
- Rank passages using TF-IDF similarity
- Display source filenames and supporting evidence
- Handle weak or missing evidence cautiously
- Log unresolved questions locally
- Evaluate retrieval against expected source filenames
- Export evaluation results as JSON
- Responsive Streamlit interface

## Technology Stack

- Python
- Streamlit
- Regular expressions and Python collections
- TF-IDF-style information retrieval
- JSON for evaluation and unresolved-question records

## Run Locally

Install Python, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit.

## How to Use

1. Launch Omnis AI.
2. Load the built-in sample or upload TXT/Markdown files.
3. Enter a question related to your documents.
4. Review the retrieved passages and source filenames.
5. Use Retrieval Evaluation to compare the retrieved source with the expected source.

## Limitations

This starter version retrieves evidence rather than generating natural-language answers with a large language model. Similarity scores are retrieval scores, not calibrated confidence probabilities. PDF parsing, semantic embeddings, persistent databases, authentication, and LLM-based answer synthesis are not included.

The unresolved-question log is stored in the application's working directory. Hosted environments may not preserve local files between restarts.

## Privacy

Do not upload confidential documents to a public demo. Review the hosting provider's data handling and access settings before using sensitive information.

## Future Improvements

- PDF and DOCX support
- Semantic embeddings and hybrid search
- LLM-generated answers with citations
- Persistent question logs
- Larger retrieval evaluation dataset
- User authentication and access controls

## Project Purpose

Built as an educational AI and information-retrieval project.

