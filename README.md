
# **Embeddings Visualizer for Language Models**
A (hopefully soon-to-be) website, API, and project for visualizing embeddings for transformer language models such as GPT, RoBERTa, BERT, BART, and more!
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
#### **Release Version v0.2 Notes:**
- Backend API version is officially at a 0.2 launch state (this is an incomplete version and there will be continuous updates).
- Assume that there may be bugs still present. One worth noting at the moment are debug calls not working with the tester application. This will be, most liklely, the next main code update.
____________________________
#### **Next update(s):**
- Start frontend considerations.
____________________________
