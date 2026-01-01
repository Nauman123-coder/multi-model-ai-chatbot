# 🤖 Multi-Model AI Chatbot

An interactive chatbot application that allows users to chat with different AI models including BlenderBot, FLAN-T5 Base, and FLAN-T5 Small.

## 🌟 Features

- **Multiple AI Models**: Choose from 3 different conversational AI models
- **Real-time Chat**: Interactive chat interface with message history
- **Model Comparison**: Switch between models to compare responses
- **Customizable Settings**: Adjust response length and other parameters
- **Beautiful UI**: Modern, gradient-based interface with smooth animations

## 🎯 Live Demo

**[Try the Live Demo Here](https://your-app-name.streamlit.app)** *(Update after deployment)*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the AI models:
```bash
python save_models.py
```
*Note: This will download ~2-3 GB of model files. It may take 10-30 minutes.*

4. Run the application:
```bash
streamlit run chatbot_app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
.
├── chatbot_app.py          # Main Streamlit application
├── save_models.py          # Script to download AI models
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore file
└── models/                # Directory for downloaded models (not in git)
    ├── blenderbot/
    ├── flan-t5-base/
    └── flan-t5-small/
```

## 🤖 Available Models

### 1. BlenderBot (Conversational)
- **Best for**: Natural, casual conversation
- **Model**: facebook/blenderbot-400M-distill
- **Size**: ~730 MB

### 2. FLAN-T5 Base (General Purpose)
- **Best for**: Instructions, Q&A, general tasks
- **Model**: google/flan-t5-base
- **Size**: ~990 MB

### 3. FLAN-T5 Small (Lightweight)
- **Best for**: Quick responses, simple tasks
- **Model**: google/flan-t5-small
- **Size**: ~308 MB

## 🎨 Screenshots

*(Add screenshots of your app here)*

![Chat Interface](screenshots/chat.png)
![Model Selection](screenshots/models.png)

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Transformers**: Hugging Face transformers library
- **PyTorch**: Deep learning framework
- **Python**: Programming language

## 📝 Usage Tips

1. **Start with BlenderBot** for natural conversations
2. **Use FLAN-T5 models** for specific questions or tasks
3. **Adjust response length** in settings for longer/shorter answers
4. **Clear chat history** to start fresh conversations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the amazing transformers library
- [Streamlit](https://streamlit.io/) for the web framework
- Facebook AI for BlenderBot
- Google Research for FLAN-T5

## 📧 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter)

Project Link: [https://github.com/your-username/your-repo-name](https://github.com/your-username/your-repo-name)

## ⚠️ Note

The models are downloaded locally when you run `save_models.py`. They are not included in the repository due to their size. For deployment on Streamlit Cloud, the models will be automatically downloaded on the first run.