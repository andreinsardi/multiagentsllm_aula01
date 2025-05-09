# Explica o agente responsável por gerar a resposta via pipeline de QA

# Importa o pipeline de QA da Hugging Face
from transformers import pipeline
from agenteA import agenteA_buscar

# Cria um pipeline de QA com modelo pré-treinado para perguntas e respostas
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define função do agente com seu comportamento principal
def agenteB_responder(pergunta: str) -> str:
    contexto = agenteA_buscar(pergunta)
    resposta = qa_pipeline(question=pergunta, context=contexto)
# Retorna o(s) bloco(s) mais relevante(s) como contexto
    return resposta["answer"]