# Intelligent Document Analyst (Round 1b)

An intelligent system that extracts and prioritizes relevant sections from PDF documents based on a given persona and job-to-be-done context. This tool is designed for Challenge 1b and provides structured outputs to support context-aware document analysis.

---

## 🧠 Key Features

- 📄 **PDF Parsing** — Extract structured outlines using font-size and content-based heading detection
- 🔍 **Semantic Ranking** — Use TF-IDF vectorization to rank document sections based on relevance
- 🧑‍💼 **Persona-Aware Filtering** — Tailor relevance scoring using domain-specific roles and tasks
- 📌 **Top Section Extraction** — Pull refined, readable content from top-ranked document sections
- 📦 **JSON Output** — Returns a detailed, timestamped report of all extracted and filtered results

---

## 🚀 How It Works

### 🔧 Step-by-step Workflow

1. **Structure Extraction**
   - Parses PDFs and identifies section headings based on font size or text patterns.
   - Creates a hierarchical outline (`H1`, `H2`, `H3`).

2. **Query Generation**
   - Builds a semantic query from the `persona` and `job_to_be_done`.

3. **TF-IDF Similarity Matching**
   - Scores and ranks document sections by comparing them with the query.

4. **Content Extraction**
   - Extracts detailed text following top-matched headings.

5. **Final Output**
   - Stores results in a structured JSON format for easy inspection and downstream use.

---

## 📂 Directory Structure

```
.
├── main.py
├── README.md
├── challenge1b_input.json
├── PDFs/
│   └── *.pdf
└── output_collection1.json
```

---

## 🔧 Installation

### 🔗 Requirements

- Python 3.7+
- [PyMuPDF](https://pypi.org/project/PyMuPDF/) (`fitz`)
- `scikit-learn`
- `nltk`
- `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pymupdf scikit-learn nltk numpy
```

Download NLTK data (automatically handled, but can be done manually):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🧪 Usage

```bash
python main.py <input_json_path> <pdf_directory> <output_json_path>
```

### Example:

```bash
python main.py "Collection 1/challenge1b_input.json" "Collection 1/PDFs" "output_collection1.json"
```

---

## 📤 Input Format (`challenge1b_input.json`)

```json
{
  "persona": {
    "role": "HR"
  },
  "job_to_be_done": {
    "task": "find onboarding processes and compliance forms"
  },
  "documents": [
    { "filename": "doc1.pdf" },
    { "filename": "doc2.pdf" }
  ]
}
```

---

## ✅ Output Format

- `metadata`: processing stats
- `extracted_sections`: top-ranked relevant headings
- `subsection_analysis`: extracted paragraph content per section

### Sample Output:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "HR",
    "job_to_be_done": "find onboarding processes and compliance forms",
    "processing_timestamp": "...",
    "total_sections_extracted": 30,
    "relevant_sections_found": 5
  },
  "extracted_sections": [...],
  "subsection_analysis": [...]
}
```

---

## 🛠 Development Notes

- Modular architecture with separate classes for:
  - `PDFStructureExtractor`
  - `SemanticAnalyzer`
  - `ContentExtractor`
- Fallback mechanisms for unreliable PDF formatting
- Custom scoring logic for relevance based on semantics and context

