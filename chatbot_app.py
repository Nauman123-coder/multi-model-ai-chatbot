"""
Multi-Model Chatbot Streamlit Application
Allows users to chat with different AI models
Optimized for Streamlit Cloud deployment - Lazy/on-demand model loading
"""
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Page configuration
st.set_page_config(
    page_title="Multi-Model Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model configurations
MODELS = {
    "BlenderBot (Conversational)": {
        "hf_model": "facebook/blenderbot-400M-distill",
        "local_path": "./models/blenderbot",
        "description": "Facebook's conversational chatbot - Best for casual conversation",
        "max_tokens": 150
    },
    "FLAN-T5 Base (General Purpose)": {
        "hf_model": "google/flan-t5-base",
        "local_path": "./models/flan-t5-base",
        "description": "Google's FLAN-T5 base model - Good for general tasks",
        "max_tokens": 150
    },
    "FLAN-T5 Small (Lightweight)": {
        "hf_model": "google/flan-t5-small",
        "local_path": "./models/flan-t5-small",
        "description": "Smaller FLAN-T5 model - Faster responses",
        "max_tokens": 150
    }
}

@st.cache_resource(show_spinner=False)
def load_model_cached(_model_key):
    """Load model and tokenizer with caching - supports both local and Hugging Face Hub"""
    model_info = MODELS[_model_key]
   
    try:
        if os.path.exists(model_info["local_path"]):
            st.info(f"Loading {model_info['hf_model'].split('/')[-1]} from local storage...")
            tokenizer = AutoTokenizer.from_pretrained(model_info["local_path"])
            model = AutoModelForSeq2SeqLM.from_pretrained(model_info["local_path"])
        else:
            st.info(f"Downloading {model_info['hf_model']} from Hugging Face... (first time only)")
            tokenizer = AutoTokenizer.from_pretrained(model_info["hf_model"])
            model = AutoModelForSeq2SeqLM.from_pretrained(model_info["hf_model"])
           
            # Save locally for future use
            os.makedirs(model_info["local_path"], exist_ok=True)
            tokenizer.save_pretrained(model_info["local_path"])
            model.save_pretrained(model_info["local_path"])
       
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model {model_info['hf_model']}: {str(e)}")
        return None, None

def generate_response(user_input, tokenizer, model, max_tokens=150):
    """Generate chatbot response"""
    try:
        inputs = tokenizer.encode(user_input, return_tensors="pt")
       
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )
       
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("🤖 Multi-Model AI Chatbot")
    st.markdown("Chat with different AI models and compare their responses!")

    if not any(os.path.exists(MODELS[m]["local_path"]) for m in MODELS):
        st.info("⏳ First time setup: Models will be downloaded automatically when first used (may take a few minutes each)")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
       
        selected_model = st.selectbox(
            "Choose AI Model",
            list(MODELS.keys()),
            help="Select which model to chat with"
        )
       
        st.info(MODELS[selected_model]["description"])
       
        with st.expander("🔧 Advanced Settings"):
            max_tokens = st.slider(
                "Max Response Length",
                min_value=50,
                max_value=300,
                value=MODELS[selected_model]["max_tokens"],
                step=10
            )
       
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.current_model = None
            st.session_state.loaded_model_key = None
            st.rerun()
       
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **BlenderBot**: Best for natural conversation  
        - **FLAN-T5 Base**: Good for instructions and Q&A  
        - **FLAN-T5 Small**: Fastest, good for simple tasks
        """)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "loaded_model_key" not in st.session_state:
        st.session_state.loaded_model_key = None  # Tracks which model is currently cached

    # Only load model when user sends a message (lazy loading)
    tokenizer = None
    model = None
    model_loaded = False

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model" in message:
                st.caption(f"*Model: {message['model']}*")

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Load model only now if not already loaded or if switched
        if (st.session_state.loaded_model_key != selected_model or 
            st.session_state.current_model != selected_model):
            
            with st.spinner(f"Loading {selected_model} model... (this may take a minute first time)"):
                tokenizer, model = load_model_cached(selected_model)
            
            st.session_state.loaded_model_key = selected_model
            st.session_state.current_model = selected_model

            if tokenizer is None or model is None:
                st.error("Failed to load the model. Please try again.")
                st.stop()
        else:
            # Model already loaded
            tokenizer, model = load_model_cached(selected_model)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, tokenizer, model, max_tokens)
            
            st.markdown(response)
            st.caption(f"*Model: {selected_model}*")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "model": selected_model
        })

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
    with col2:
        st.metric("Current Model", selected_model.split("(")[0].strip())
    with col3:
        st.metric("Max Tokens", max_tokens)

if __name__ == "__main__":
    main()
