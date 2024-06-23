# Medical Chatbot Project

This repository contains code and resources for a chatbot project. The project leverages natural language processing (NLP) techniques and machine learning models to build an intelligent chatbot that can sufficiently 
respond to a patient's prompts.

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/kuriofoolio/CustomChatbot.git
    cd CustomChatbot
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK data**:
    ```bash
    python nltk_data.py
    ```

## Usage

- **Run the chatbot application**:
    ```bash
    python chatbot_app.py
    ```

- **Preprocess data**:
    ```bash
    python data_preprocess.py
    ```

- **Train the model**:
    Modify and use `main.py` as needed to train your chatbot model.

## Contributing

Please feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
