import openai
from dotenv import dotenv_values
from src.vector_db import PineconeConnector
import gradio as gr

ENV_VARS = dotenv_values()

# Initialize pinecone API variables
pinecone_api_key = ENV_VARS["PINECONE_API_KEY"]
pinecone_env = "us-west4-gcp"
index_name = "ec-decisions-test"
pine = PineconeConnector(index_name, pinecone_api_key, pinecone_env)
pine.index.describe_index_stats()

# Anitialize openai API key and embed model
openai_api_key = ENV_VARS["OPENAI_API_KEY"]
openai.api_key = openai_api_key
embed_model = "text-embedding-ada-002"

def embed_query(query):
    # Embed query
    r = openai.Embedding.create(input=query, engine=embed_model)
    embeds = [record['embedding'] for record in r['data']]
    return embeds

def find_contexts(embeds):
    pine_search = pine.index.query(embeds, top_k=5, include_metadata=True)
    # Return context and source
    contexts = [f"{i['metadata']['text']} (source: {i['metadata']['text']})" for i in pine_search["matches"]]
    return contexts

def primer_prompt():
    primer = """
        You are Q&A bot who is an economic expert. A highly intelligent system that answers
        user questions based on the information provided by the user above each question. 
        If the information can not be found in the information provided by the user you truthfully say "I don't know". 
        Don't mention that you answer based on the information provided but do mention the source.
    """
    return primer

def ask_leibniz(query):
    embeds = embed_query(query)
    contexts = find_contexts(embeds)
    primer = primer_prompt()

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": "\n\n---\n\n".join(contexts)+"\n\n-----\n\n" + query}
        ]
    )
    return answer['choices'][0]['message']['content']

def gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            ## Ask Gottfried 
            Hello! I am a Q&A chatbot that can help you retrieve information from a database of EC's documents.
            I don't know much yet, but you can ask me about:
            * [Summary of Commission Decision: Case AT.40099 — Google Android](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52019XC1128(02)&from=EN)
            * [Case M.8322 – HEINEKEN UK / PUNCH TAVERNS SECURITISATION](https://ec.europa.eu/competition/mergers/cases/decisions/m8322_124_3.pdf)
            * [Digital Markets Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R1925&from=EN)
            """ 
        )
            
        with gr.Tab("Chat"):
            inputs = gr.Textbox(lines=2, placeholder="Your question here...", label="Question")
            outputs =gr.Textbox(label="Answer",placeholder="I am quite new so I don't know everything.")
            text_button = gr.Button("Submit")
        with gr.Tab("How do I work?"):
            gr.Markdown(
                """
                > Here is how I work:
                First, I load selected European Commissions' documents and split them into smaller chunks. Then, I use OpenAI embeddings (*ada-002*) to convert each chunk into a numerical representation, which I store in Pinecone, a vector database.
                When you ask me a question, I use the prompt to search for the most similar chunks in the Pinecone database. This helps me quickly find relevant information from the documents.
                Next, I send the prompt to the OpenAI API, which uses a LLM (*GPT-3.5-turbo*) to provide an answer based on the most relevant chunks retrieved from Pinecone. The answer is then sent back to you as a response to your question.
                """
            )

        text_button.click(fn=ask_leibniz, inputs=inputs, outputs=outputs)

    # demo.launch(auth=("gottfried", "leibniz"), share=True)
    demo.launch()