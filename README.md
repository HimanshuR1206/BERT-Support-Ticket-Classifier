# BERT Support Ticket Classifier

A Natural Language Processing project that uses a fine-tuned BERT model to automatically classify customer support tickets.

## Categories

The model classifies tickets into four categories:

- Billing
- Sales & Marketing
- Service Issues
- Technical Support

## Model Performance

Final test results:

- Accuracy: 99.58%
- Precision: 99.59%
- Recall: 99.58%
- Weighted F1-score: 99.58%
- Test samples: 240
- Correct predictions: 239/240

The reported performance was measured on a cleaned, balanced, high-confidence four-class evaluation dataset.

## Dataset

The dataset contains 1,600 support tickets.

- Training: 1,120
- Validation: 240
- Test: 240
- Four categories
- 400 examples per category

The model uses the ticket subject and body as input.

## Tech Stack

- Python
- BERT
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Streamlit
- Pandas
- NumPy

## Model

Base model:

bert-base-uncased

Training configuration:

- Epochs: 5
- Learning rate: 2e-5
- Training batch size: 16
- Evaluation batch size: 32
- Maximum sequence length: 256
- Weight decay: 0.01

## How It Works

Customer Ticket
-> Subject + Description
-> BERT Tokenizer
-> Fine-tuned BERT
-> Classification Layer
-> Predicted Support Category

## Installation

Install the dependencies with:

    pip install -r requirements.txt

## Run the Application

Start Streamlit with:

    streamlit run app.py

## Example

Subject:

    Payment failed

Description:

    My credit card is being declined when I try to renew my subscription.

Expected category:

    Billing

## Project Structure

    BERT-Support-Ticket-Classifier/
    |
    |-- app.py
    |-- README.md
    |-- requirements.txt
    |-- .gitignore
    |
    |-- data/
    |
    `-- notebooks/

## Future Improvements

- Train on a larger manually verified dataset
- Add more support categories
- Add multilingual classification
- Deploy the application online
- Add REST API support
- Add confidence-based human escalation

## Disclaimer

The reported model performance applies to the cleaned and optimized evaluation dataset used in this project and should not be interpreted as guaranteed performance on arbitrary real-world support tickets.
