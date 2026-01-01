"""
Multi-Model Chatbot Streamlit Application
Allows users to chat with different AI models
Optimized for Streamlit Cloud deployment
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

@st.cache_resource
def load_model(model_name):
    """Load model and tokenizer with caching - supports both local and Hugging Face Hub"""
    model_info = MODELS[model_name]
    
    # Try loading from local path first, fallback to Hugging Face Hub
    try:
        if os.path.exists(model_info["local_path"]):
            st.info(f"Loading {model_name} from local storage...")
            tokenizer = AutoTokenizer.from_pretrained(model_info["local_path"])
            model = AutoModelForSeq2SeqLM.from_pretrained(model_info["local_path"])
        else:
            st.info(f"Downloading {model_name} from Hugging Face Hub... (first time only)")
            tokenizer = AutoTokenizer.from_pretrained(model_info["hf_model"])
            model = AutoModelForSeq2SeqLM.from_pretrained(model_info["hf_model"])
            
            # Save locally for future use
            os.makedirs(model_info["local_path"], exist_ok=True)
            tokenizer.save_pretrained(model_info["local_path"])
            model.save_pretrained(model_info["local_path"])
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_response(user_input, tokenizer, model, max_tokens=150):
    """Generate chatbot response"""
    try:
        # Tokenize input
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=max_tokens,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    # Header
    st.title("🤖 Multi-Model AI Chatbot")
    st.markdown("Chat with different AI models and compare their responses!")
    
    # Add info about deployment
    if not any(os.path.exists(MODELS[m]["local_path"]) for m in MODELS):
        st.info("⏳ First time setup: Models will be downloaded automatically (may take 5-10 minutes)")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose AI Model",
            list(MODELS.keys()),
            help="Select which model to chat with"
        )
        
        # Display model info
        st.info(MODELS[selected_model]["description"])
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            max_tokens = st.slider(
                "Max Response Length",
                min_value=50,
                max_value=300,
                value=MODELS[selected_model]["max_tokens"],
                step=10,
                help="Maximum number of tokens in response"
            )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **BlenderBot**: Best for natural conversation
        - **FLAN-T5 Base**: Good for instructions and Q&A
        - **FLAN-T5 Small**: Fastest, good for simple tasks
        """)
        
        # GitHub link
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[📚 GitHub Repository](https://github.com/your-username/your-repo)")
        st.markdown("[⭐ Star on GitHub](https://github.com/your-username/your-repo)")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    
    # Check if model changed
    if st.session_state.current_model != selected_model:
        st.session_state.current_model = selected_model
        if st.session_state.messages:
            st.info(f"Switched to {selected_model}. Previous conversation is still shown.")
    
    # Load selected model
    with st.spinner(f"Loading {selected_model}..."):
        tokenizer, model = load_model(selected_model)
    
    if tokenizer is None or model is None:
        st.error("Failed to load model. Please refresh the page and try again.")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model" in message:
                st.caption(f"*Model: {message['model']}*")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, tokenizer, model, max_tokens)
            
            st.markdown(response)
            st.caption(f"*Model: {selected_model}*")
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "model": selected_model
        })
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Current Model", selected_model.split()[0])
    with col3:
        st.metric("Max Tokens", max_tokens)

if __name__ == "__main__":
    main()