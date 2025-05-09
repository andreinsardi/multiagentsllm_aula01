from langchain.agents import Tool, initialize_agent
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, set_seed
from agenteA import agenteA_buscar
from agenteB import agenteB_responder
from agenteC import agenteC_avaliar

# Define ferramentas como agentes
tools = [
    Tool(name="ContextRetriever", func=agenteA_buscar, description="Recupera contexto de Dom Casmurro com base na pergunta."),
    Tool(name="AnswerGenerator", func=agenteB_responder, description="Gera uma resposta com base no contexto recuperado."),
    Tool(name="AnswerEvaluator", func=agenteC_avaliar, description="Avalia se a resposta gerada está boa ou precisa ser refeita.")
]

# Inicializa LLM local com huggingface
generator = pipeline("text-generation", model="gpt2", max_new_tokens=64, do_sample=True)
llm = HuggingFacePipeline(pipeline=generator)

# Cria o agente com as ferramentas
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)

# Executa o agente
if __name__ == "__main__":
    pergunta = "Quem era Capitu?"
    resposta = agent.run(pergunta)
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta}")
