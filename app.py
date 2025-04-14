from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import streamlit as st
from io import BytesIO
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
import ollama
import os
from evaluation import HallucinationEvaluator, CustomOllamaMistralModel


embeddings = []

##Data Extraction from Website
loader = WebBaseLoader("https://www.ncbi.nlm.nih.gov/books/NBK221839/")
health_data = loader.load()[0].page_content.split("\n\n\n\n"*10)[0].replace("  ","").replace("\n","").replace("\t","")
health_token =  [health_data[i:i+1000] for i in range(0,len(health_data),1000)]
with open("Rules data 2.txt", "r", encoding="utf-8") as file:
    # Read the contents of the file
    content = file.read()
full_chuck = content.split("##")
final_chuck = health_token + full_chuck


st.set_page_config(page_title="Your App Title", layout="wide")
# App title
st.title("🧘‍♀️ Health & Wellness Coach - Fitness Goal Generator")

# Instructions
st.write("""
Upload a CSV or Excel file with a single client's health data. Then, update the fitness goal and download the updated profile.
""")

# File uploader (supports CSV or Excel)
uploaded_file = st.file_uploader("📁 Upload your client data (CSV/Excel)", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        # Read file based on file type
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)


            # Display the data
            st.subheader("📄 Client Information")
            st.dataframe(df)

            # Prompt for fitness goal update
            st.subheader("🎯 Enter or Update Fitness Goal")
            default_goal = df['Health Goal'].values[0] if 'Health Goal' in df.columns else ''
            # Create two columns: one for the input and the other for the model selector
            col1, col2 = st.columns([3, 1])  # Adjust column width ratio as needed
            with col1:
                # Display the fitness goal input in col1
                new_goal=st.text_input("Enter fitness goal for the client:", value=default_goal, key="goal_input")
            with col2:
                # Display the model selector on the right side in col2
                selected_model = st.selectbox("Choose a model:", ["mistral","llama3.2", "gemma3"], key="model_select")

            # Initialize the LLM with the selected model
            llm = OllamaLLM(model=selected_model)
            # Button to generate fitness goal
            if st.button("Generate Fitness Goal"):
                if new_goal:
                    # Update the fitness goal in the data
                    
                    embeddings = []
                    if os.path.exists("my_index.faiss"):
                        print("********")
                        index = faiss.read_index("my_index.faiss")
                    else:
                        for text in final_chuck:
                            response=ollama.embeddings(model="nomic-embed-text",prompt=text)
                            embeddings.append(response["embedding"])
                        dimension = len(embeddings[0])

                        index = faiss.IndexFlatL2(dimension)
                        index.add(np.array(embeddings).astype("float32"))

                        faiss.write_index(index,"my_index.faiss")

                    query = new_goal
                    query_vec = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
                    D, I = index.search(np.array([query_vec]).astype("float32"), k=10)
                    doc1 = []
                    for i in I[0]:
                        doc1.append(final_chuck[i])
                    doc_str = str(doc1)
                    template = """
                    Develop a Health Advisor AI that generates actionable, evidence-based recommendations for
                    achieving user-defined health goals ({new_goal}). Utilize user's personal data ({data}) and reliable resources
                    ({doc_str}) to provide concise, tailored advice focused on optimizing results. Your response should be Grounded. It should be Eye cathching and Jazzy"""
                    prompt = ChatPromptTemplate.from_messages([
                        ('system', "I am expert Medical, Fitness Coach/Advisor"),
                        ('human', f"{template}")
                    ])

                    chain = prompt|llm
                    print(chain)
                    result = chain.invoke({"new_goal": new_goal, "data": df, "doc_str":{doc_str}})
                    print(result)
                    st.success("✅ Fitness goal generated successfully!")
                    st.write(f"**{result}**")

                    hallucination_evaluator = HallucinationEvaluator(
                        context=doc1,
                        prompt=new_goal,
                        actual_output=result,
                        model_name=selected_model
                    )
                    hallucination_result = hallucination_evaluator.hallucination()
                    contextual_relavancy_result = hallucination_evaluator.contextual_relavancy()
                    answer_relevancy_result = hallucination_evaluator.answer_relevancy()
                    # Create Dashboard for evaluation results
                    st.subheader("📊 Evaluation Results")
                    st.write("**Hallucination Metrics: 0 is good, 1 is bad**")
                    st.write("**Hallucination Score:**", hallucination_result["hallucination_score"])
                    # st.write("**Contextual Precision Score:**", contextual_relavancy_result["contextual_relavancy_score"])
                    # st.write("**Answer Relevancy Score:**", answer_relevancy_result["answer_relevancy_score"])

                    
                else:
                    st.warning("⚠️ Please enter a fitness goal.")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("📌 Please upload a CSV or Excel file to get started.")


# response = llm.invoke(prompt)
# print(response)
