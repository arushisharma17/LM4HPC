from sentence_transformers import SentenceTransformer

# Load the model
model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Save the model locally
local_path = '$PSCRATCH'
model.save(local_path)

