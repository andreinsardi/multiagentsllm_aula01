# Importa o agente responsável por recuperar o contexto mais relevante usando embeddings
from agenteA import agenteA_buscar
# Importa o agente responsável por gerar uma resposta com base no contexto
from agenteB import agenteB_responder
# Importa o agente que avalia a qualidade da resposta gerada
from agenteC import agenteC_avaliar


# Função principal que coordena os três agentes
def executar_fluxo(pergunta):
    # Exibe a pergunta feita pelo usuário
    print(f"[Usuário]: {pergunta}")
    
    # Recupera o contexto textual mais relevante com base na pergunta
    contexto = agenteA_buscar(pergunta)
    # Informa o tamanho do contexto recuperado
    print(f"[AgenteA - ContextRetriever]: Contexto recuperado com {len(contexto)} caracteres")

    # Gera uma resposta usando o modelo de QA da Hugging Face
    resposta = agenteB_responder(pergunta)
    # Exibe a resposta gerada pelo agente B
    print(f"[AgenteB - AnswerGenerator]: {resposta}")

    # Avalia se a resposta gerada é suficientemente boa ou precisa ser refeita
    avaliacao = agenteC_avaliar(resposta)
    # Mostra a avaliação final da resposta
    print(f"[AgenteC - AnswerEvaluator]: {avaliacao}")


# Executa o fluxo de agentes quando o script é executado diretamente
if __name__ == "__main__":
    executar_fluxo("Quem era Capitu?")