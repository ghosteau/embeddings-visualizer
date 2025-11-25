
# **Embeddings Visualizer for Language Models**
A website and project for visualizing embeddings for transformer language models such as GPT, RoBERTa, BERT, BART, and more!
___________________________
### What is this project?
This project provides an interactive platform for exploring and understanding token embeddings from transformer-based language models. It loads models like GPT-2, RoBERTa, BERT, and others, extracts their embedding spaces, and visualizes relationships between tokens using dimensionality-reduction techniques. The goal is to make it easy for researchers, students, and developers to inspect how language models represent meaning, similarity, and structure in high-dimensional vector spaces.
___________________________
## **Contributor(s):**
- Manny McGrail -- Project Management, Backend API, and ML Module Management
____________________________
## **Necessary Files:**


**Requirements file to download for dependencies**: [requirements.txt](requirements.txt)

*Download the full requirements file using*: `pip install -r requirements.txt`

__________________________

### **Implementation Details:**

Backend:
- Python
    - FastAPI
- Other dependencies:
  - Hugging Face


Frontend:
- JavaScript
  - React
  - ThreeJS
____________________________
#### **Release Version v1.0 Notes:**
- Backend API version is officially at a 1.0 launch state.
____________________________
#### **Next update(s):**
- Check over unit tests and code review backend.
- Clean up notebooks for readability. 
  - Break up into blocks and remove unnecessary comments.
- Split code up accordingly.
  - For instance, running server should be main.
  - Distance calculations in one class, so on and so forth.
- Make sure .gitignore is up-to-date and ignoring all necessary files.
- Make README thorough, professional, and clean looking.
- Start frontend considerations.
____________________________