
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(
    page_title="BERT Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

HF_MODEL = "YOUR_HF_USERNAME/bert-support-ticket-classifier"

id2label = {
    0: "Billing",
    1: "Sales & Marketing",
    2: "Service Issues",
    3: "Technical Support"
}

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(HF_MODEL)

    model = BertForSequenceClassification.from_pretrained(
        HF_MODEL
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

def predict_ticket(subject, body):
    text = subject.strip() + " " + body.strip()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_id].item()

    return (
        id2label[predicted_id],
        confidence,
        probabilities.cpu().numpy()
    )

st.title("🎫 BERT Support Ticket Classifier")

st.write(
    "Classify customer support tickets using a fine-tuned BERT model."
)

subject = st.text_input(
    "Ticket Subject",
    placeholder="Example: Payment failed"
)

body = st.text_area(
    "Ticket Description",
    placeholder="Describe the issue...",
    height=180
)

if st.button(
    "Classify Ticket",
    type="primary",
    use_container_width=True
):

    if not subject.strip() or not body.strip():
        st.warning("Please enter subject and description.")

    else:
        category, confidence, probabilities = predict_ticket(
            subject,
            body
        )

        st.success(f"Predicted Category: {category}")

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Category Probabilities")

        for i, probability in enumerate(probabilities):
            st.write(
                f"**{id2label[i]}: {probability * 100:.2f}%**"
            )
            st.progress(float(probability))

st.caption(
    "Fine-tuned BERT model for support ticket classification."
)
