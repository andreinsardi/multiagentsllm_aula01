import re
# Biblioteca pandas para manipulação de dados tabulares
import pandas as pd
# Importa modelo de embeddings semânticos (SentenceTransformer)
from sentence_transformers import SentenceTransformer
# Biblioteca de indexação vetorial eficiente (HNSW)
import hnswlib

# Leitura do texto de entrada
with open("dom_casmurro.txt", encoding="latin1") as f:
    texto = f.read()

# Inicializa modelo de embeddings
# Carrega o modelo de embedding 'all-MiniLM-L6-v2'
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Divide o texto em blocos a partir dos capítulos
blocos = re.split(r"(?i)cap[íi]tulo\s+\d+", texto)
blocos = [b.strip() for b in blocos if len(b.strip()) > 100]

# Gera os embeddings
# Gera embedding da pergunta ou bloco textual
embeddings = modelo.encode(blocos, convert_to_numpy=True)

# Indexa os embeddings com hnswlib
# Cria o índice de busca baseado em vetores (HNSW)
index = hnswlib.Index(space="cosine", dim=384)
# Inicializa a estrutura do índice HNSW
index.init_index(max_elements=len(blocos), ef_construction=200, M=16)
# Adiciona os vetores de embeddings ao índice
index.add_items(embeddings)

# Salva o índice e os blocos
# Salva o índice vetorial para uso posterior
index.save_index("indice.bin")
# Salva os blocos em um CSV
pd.DataFrame({"bloco": blocos}).to_csv("blocos.csv", index=False)

print("Indexação finalizada com sucesso.")