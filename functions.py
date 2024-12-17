import panns_inference
import librosa
import numpy as np

# Create a Milvus client connection
client = MilvusClient("./milvus_demo.db")
# Create the collection with the correct embedding dimension (2048)
def create_audio_collection():
    client.create_collection(
        collection_name="audio_panns_collection",
        dimension=2048,  
    )
    print("Audio collection created.")


# Extract embedding using PANNs model
def extract_audio_embedding_panns(audio_file_path):
    at = panns_inference.AudioTagging(checkpoint_path=None, device='cpu')
    audio, _ = librosa.load(audio_file_path, sr=32000, mono=True)
    audio = audio[None, :]  # Add batch dimension
    clipwise_output, embedding = at.inference(audio)
    return embedding.flatten()  # Flatten the embedding to a 1D array


# Insert audio embeddings into Milvus
def insert_audio_data(audio_files):
    data = []
    for i, audio_file_path in enumerate(audio_files):
        # Extract audio embedding using PANNs
        embedding = extract_audio_embedding_panns(audio_file_path)
        print(f"Embedding size for {audio_file_path}: {embedding.shape}")

        # Audio metadata (for example, file path and duration)
        audio_metadata = {
            "audio_file_path": audio_file_path,
            "duration": librosa.get_duration(filename=audio_file_path)  # Get duration from librosa
        }

        # Prepare data for insertion into Milvus
        data.append({
            "id": i,
            "vector": embedding.tolist(),  # Convert numpy array to list for insertion
            "audio_file_path": audio_metadata["audio_file_path"],
            "duration": audio_metadata["duration"]
        })

    # Insert data into Milvus
    res = client.insert(
        collection_name="audio_panns_collection",
        data=data,
    )
    print(f"Inserted {len(audio_files)} audio embeddings.")
    return res


# Search for similar audio to the query audio
def search_similar_audio(query_audio_file):
    # Extract embedding for the query audio file
    query_embedding = extract_audio_embedding_panns(query_audio_file)
    print(f"Query embedding size: {query_embedding.shape}")

    # Perform a vector search on the Milvus collection
    search_params = {
        "metric_type": "L2",  # Use L2 (Euclidean) distance for similarity search
        "params": {"nprobe": 10}  # Number of probes, increase for more accuracy (higher search time)
    }

    # Perform the search
    res = client.search(
        collection_name="audio_panns_collection",  # Name of the collection
        data=[query_embedding.tolist()],  # Query vector (converted to list)
        top_k=3,  # Number of similar items to retrieve
        params=search_params,
        output_fields=["audio_file_path"]  # Get the file path of the audio
    )

    # Display the results (most similar audio)
    print(f"Top 3 most similar audio files to {query_audio_file}:")
    for idx, result in enumerate(res[0]):
        # Check the available fields in the result
        print(f"Result {idx + 1}: {result}")
        # Assuming the fields are part of the dictionary, adjust accordingly
        audio_file_path = result.get('audio_file_path', 'N/A')
        distance = result.get('distance', 'N/A')
        print(f"Rank {idx+1}: {audio_file_path} with distance {distance}")


  
