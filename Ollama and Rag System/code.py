import streamlit as st
from langchain.llms import Ollama

def load_model():
    return Ollama(model="mistral")


def generate_response(model, operation, user_input):
    prompts = {
        "Generate Text": f"Generate a creative response: {user_input}",
        "Translate Text": f"Translate this to Hindi: {user_input}",
        "Summarize Text": f"Summarize this text: {user_input}"
    }
    
    return model(prompts[operation])

def main():
    st.title("📝 AI-Powered Text Assistant")
    st.subheader("Generate, Translate,  and Summarize ")

   
    model = load_model()

    operation = st.selectbox("Choose an Operation", ["Generate Text", "Translate Text", "Summarize Text"])

    user_input = st.text_area("Enter your text:", height=200)
    if st.button("Run AI"):
        if user_input.strip():  
            response = generate_response(model, operation, user_input)
            st.subheader(f"📝 {operation} Result:")
            st.write(response)
        else:
            st.warning("Please enter some text.")


if __name__ == "__main__":
    main()
