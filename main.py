import ollama
from vector_db import init_collection, add_chunk_to_database, client, COLLECTION_NAME, create_embedding

dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3.2:3b'

# Initialize Qdrant collection
init_collection()
print(f'Initialized Qdrant collection: {COLLECTION_NAME}')


def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
  query_embedding = create_embedding(query)
  
  # Search in Qdrant using query_points
  search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=top_n
  )
  
  # Extract chunks and scores
  results = [(point.payload["text"], point.score) for point in search_result.points]
  return results


for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')


input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print(retrieved_knowledge)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

out = {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])} 
instruction_prompt = f'''You are a helpful chatbot.  Use only the following pieces of context to answer the question. Don't make up any new information: {out}'''

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
