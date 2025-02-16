import streamlit as st
from langchain_community.llms import Ollama


def load_model():
    return Ollama(model="mistral")
 
# for only fashion 
def generate_response(model, user_query):
    fashion_keywords = ["fashion", "outfit", "style", "clothing", "dress", "shoes", "accessory", "trend", "color", "matching", "wear"]

 
    if not any(keyword in user_query.lower() for keyword in fashion_keywords):
        return "❌ Sorry, fashion is my expertise. I can't answer that."

  
    prompt = f"""You are a professional fashion assistant. Answer only fashion-related questions with expert advice.
    
    Question: {user_query}

    Provide a specific response based on the latest fashion trends, outfit coordination, and personal styling.
    """

    print("Prompt Sent to Model:", prompt) 
    return model(prompt)

def main():
    st.title("👗 Fashion AI Assistant")
    st.subheader("💡 Get expert fashion advice and recommendations!")
    model = load_model()
    user_query = st.text_area("Ask me anything about fashion:", height=100)
    if st.button("Get Advice"):
        if user_query.strip():
            response = generate_response(model, user_query)
            st.subheader("📖 Fashion Advice:")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a fashion-related question.")

if __name__ == "__main__":
    main()
