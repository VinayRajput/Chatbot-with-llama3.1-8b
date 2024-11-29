# README.md for Configuring Chatbot with Llama 3.1:8B on Local Machine

This document provides a step-by-step guide to install and run the Llama 3.1:8B model on your local machine and configure a chatbot which can read document file and answer questions.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python** (version 3.8 or higher)
- **Pip** (Python package installer)
- **Git** (for cloning repositories)
- **VSCode** (recommended for editing and debugging code)

## Installation Steps

### 1. Download Ollama Toolkit

**Ollama** is required to run Llama models. Follow these steps based on your OS:

1. Download the Ollama installer from the official Ollama website.
2. Run the installer and follow the on-screen instructions.
3. Verify installation by opening  Terminal / PowerShell and running:
   ```bash
   ollama --version 
   ```
   
### 2. Install & run Llama 3.1:8B
###  _**Check here before proceeding it require 4.7gb of internet data download and disk space**_

[https://ollama.com/library/llama3.1:8b](https://ollama.com/library/llama3.1:8b)


1. **Open Terminal or Command Prompt** based on your OS.
2. **Run the following command** to install and run the Llama 3.1:8B model locally:
   ```bash
   ollama run llama3.1:8b
   ```
   
### 3. Clone this repository
   ```bash
   git clone git@github.com:VinayRajput/Chatbot-with-llama3.1-8b.git
   ```
# Required Python Packages

## Install using pip
### langchain
```bash
pip install --upgrade --quiet langchain-openai langchain
pip install streamlit
```


### langchain_community
```bash
pip install python-dotenv langchain-community
```
### langchain_chroma
```bash
pip install -U langchain_chroma
```

### langchain_ollama
```bash
pip install -U langchain-ollama
```

### callbacks
```bash
pip install langchain-community
```

# Optional Python Packages

## Install using pip
### PyPDFLoader
```bash
pip install pypdf
```

# Run Python Code
streamlit run app.py
