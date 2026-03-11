# 🇮🇳 Bharat Sahayak AI
### Multi-Agent AI System for Government Scheme Assistance

Bharat Sahayak AI is a **multi-agent AI platform** designed to help citizens understand, access, and apply for Indian government welfare schemes.

The system uses **Natural Language Processing, Retrieval Augmented Generation (RAG), Document AI, and Speech AI** to guide users through the entire process — from asking a question to submitting an application.

This project is implemented as a **collaborative multi-agent architecture**, where each agent performs a specialized AI task.

---

# 🚀 Features

### 🔹 Natural Language Understanding
Users can describe their problems in **Hindi, English, or Hinglish**, and the system automatically understands:

- user intent  
- scheme category  
- key entities like age, income, location  

---

### 🔹 Scheme Recommendation Engine
The system recommends the **most relevant government schemes** using:

- semantic search
- vector embeddings
- RAG-based AI answers

---

### 🔹 AI-powered Question Answering
Users can ask questions like:

```
"Meri mummy ko pension nahi mil rahi"
"Am I eligible for Ayushman Bharat?"
"Kisan ke liye kaunsi scheme hai?"
```

The AI retrieves scheme documents and generates accurate answers.

---

### 🔹 Document Understanding
Users can upload documents such as:

- Aadhaar card
- PAN card
- Income certificate

The system extracts relevant fields and **auto-fills government forms**.

---

### 🔹 Voice Interaction
The platform supports **voice input and output** for accessibility.

Users can speak their problem, and the AI responds in natural speech.

---

### 🔹 Grievance Monitoring
The system predicts:

- expected resolution time
- escalation recommendations

using ML models.

---

# 🧠 System Architecture

The system is designed as a **multi-agent pipeline** where each agent handles a specific task.

```
User Input (Voice / Text)
        │
        ▼
Voice Agent (Speech → Text)
        │
        ▼
Intent Recognition Agent
(Intent Classification + NER)
        │
        ▼
RAG Scheme Recommendation Agent
(Vector Search + LLM)
        │
        ▼
Form Understanding Agent
(Document OCR + Auto Fill)
        │
        ▼
Monitoring Agent
(Prediction + Escalation)
        │
        ▼
User Dashboard
```

---

# 👥 Team Structure

This project is implemented by **4 members**, each responsible for a specialized AI module.

---

## 👤 Member 1 — Intent & Entity Recognition Agent

### Responsibilities
- Intent classification using **DistilBERT**
- Named Entity Recognition using **spaCy**
- Urgency classification
- Text preprocessing for Hinglish input

### Output

```
User Query → {
   intent: "pension",
   urgency: "high",
   entities: {
       age: 65,
       location: "Uttar Pradesh"
   }
}
```

---

## 👤 Member 2 — Scheme Recommendation & RAG Agent

### Responsibilities
- Convert government scheme documents into embeddings
- Store embeddings in **FAISS vector database**
- Implement semantic search
- Build RAG-based question answering system
- Implement scheme eligibility checker

### Technologies

- Sentence Transformers
- FAISS Vector DB
- LangChain
- Google Gemini API
- Decision Tree (scikit-learn)

---

## 👤 Member 3 — Form Understanding Agent

### Responsibilities
- Document type classification
- OCR-based information extraction
- Automatic form filling
- Confidence scoring

### Technologies

- MobileNet / ResNet
- Tesseract OCR
- OpenCV
- Logistic Regression

---

## 👤 Member 4 — Voice Interface & Monitoring Agent

### Responsibilities

- Speech to Text
- Text to Speech
- Resolution time prediction
- Escalation recommendation

### Technologies

- OpenAI Whisper
- gTTS
- Random Forest
- XGBoost

---

# 📂 Project Structure

```
bharat-sahayak-ai/
│
├── models/
│   ├── config.json
│   └── training_args.bin
│
├── src/
│   ├── loader.py
│   ├── preprocess.py
│   ├── extractor.py
│   ├── main_graph.py
│
│   ├── rag_agent.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── qa_generator.py
│   └── eligibility_checker.py
│
├── data/
│   └── schemes/
│
├── tests/
│   └── test_agents.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The RAG system uses **government scheme documents**.

Example schemes included:

- Indira Gandhi National Old Age Pension Scheme
- Ayushman Bharat PM-JAY
- PM Kisan Samman Nidhi
- Public Distribution System
- PM Awas Yojana
- PM Kaushal Vikas Yojana
- PM Suraksha Bima Yojana
- PM Matru Vandana Yojana

Documents are converted into **vector embeddings** for semantic retrieval.

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/your-username/bharat-sahayak-ai.git
```

Move into the project folder:

```
cd bharat-sahayak-ai
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the main pipeline:

```
python src/main_graph.py
```

---

# 🧪 Running Tests

```
pytest tests/
```

---

# 🛠 Technologies Used

- Python
- LangGraph
- HuggingFace Transformers
- Sentence Transformers
- FAISS
- LangChain
- Google Gemini API
- Tesseract OCR
- Whisper Speech Model
- Scikit-learn

---

# 📌 Future Improvements

- Multilingual support for Indian languages
- Real-time government API integration
- Mobile application interface
- Advanced document verification

---

# 📄 License

This project is intended for **academic and research purposes**.

---

# ⭐ Acknowledgements

We acknowledge the use of **public government scheme data** and open-source AI tools that made this project possible.
