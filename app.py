import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import numpy as np
from gnn_model import GNNBinaryClassifier  # GCN model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Model Loading with Caching (Streamlit) --
@st.cache_resource
def load_text_model():
    # Load fine-tuned stance detection model (e.g., DeBERTa-based)
    model = AutoModelForSequenceClassification.from_pretrained("models/stance_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("models/stance_model")
    return model, tokenizer

@st.cache_resource
def load_clip_model():
    # Load CLIP for image-text similarity
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def load_gnn_model():
    # Initialize GNN and load weights
    model = GNNBinaryClassifier(in_channels=4, hidden_channels=16, num_classes=2)
    model.load_state_dict(torch.load("models/gnn_misinformation.pt", map_location=device))
    model.to(device).eval()
    return model

# -- Utility Function to Qualify Confidence Levels --
def label_confidence(score):
    if score >= 0.7:
        return "🟢 HIGH"
    elif score >= 0.3:
        return "🟠 MODERATE"
    else:
        return "🔴 LOW"

# -- Prediction Functions --
def predict_stance(text):
    # Tokenize and run text model
    inputs = stance_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = stance_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_label].item()
    return ("real" if pred_label == 1 else "fake"), confidence

def predict_clip(image, caption):
    # Preprocess and run CLIP
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # Compute cosine similarity between image and text embeddings
        img_emb = outputs.image_embeds.cpu().numpy()
        txt_emb = outputs.text_embeds.cpu().numpy()
        sim = float(cosine_similarity(img_emb, txt_emb)[0,0])
    label = "real" if sim > 0.3 else "fake"  # threshold 0.3 (heuristic)
    return label, sim

def predict_with_gnn(graph_data):
    # Load GNN model and run on graph data
    model = load_gnn_model()
    graph_data = graph_data.to(device)
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
    # Assume a single node or unified label per graph
    pred_node = out.argmax(dim=1)
    # If multiple nodes, this picks the first; customize if needed
    final_pred = pred_node[0].item() if pred_node.numel() > 0 else 0
    return "real" if final_pred == 1 else "fake"

def ensemble_with_gnn(text_pred, text_conf, clip_pred, clip_conf, gnn_pred, alpha=0.4, beta=0.3):
    # Convert labels to scores (1 for "real", 0 for "fake")
    t_score = 1 if text_pred == "real" else 0
    c_score = 1 if clip_pred == "real" else 0
    g_score = 1 if gnn_pred == "real" else 0
    score = alpha * text_conf * t_score + beta * clip_conf * c_score + (1 - alpha - beta) * g_score
    return ("real" if score >= 0.5 else "fake"), round(score, 3)

# -- Graph Visualization Helpers --
def visualize_graph(graph_data):
    G = nx.Graph()
    edge_list = graph_data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    fig, ax = plt.subplots()
    nx.draw(G, ax=ax, with_labels=True)
    st.pyplot(fig)

def visualize_tsne(model, graph_data):
    # Get node embeddings from first GCN layer
    model.eval()
    with torch.no_grad():
        embeds = model.conv1(graph_data.x, graph_data.edge_index).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)
    labels = graph_data.y.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in np.unique(labels):
        idx = labels == label
        ax.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=f"Class {label}", alpha=0.7)
    ax.legend()
    ax.set_title("GCN Node Embeddings (t-SNE)")
    st.pyplot(fig)

# -- Load models once --
stance_model, stance_tokenizer = load_text_model()
clip_model, clip_processor = load_clip_model()

# -- Streamlit UI --
if __name__ == "__main__":
    st.title("Disaster Misinformation Detector (Multimodal)")
    st.markdown("Analyze a disaster-related tweet using **text**, **image**, and **graph** data.")

    tweet = st.text_area("📝 Tweet Text")
    image_file = st.file_uploader("📷 Upload Disaster Image (optional)", type=["jpg", "jpeg", "png"])
    graph_file = st.file_uploader("📈 Upload Graph Data (.pt, optional)", type=["pt"])

    if st.button("Run Misinformation Check"):
        if not tweet:
            st.warning("Please enter tweet text.")
        else:
            # Text prediction
            text_pred, text_conf = predict_stance(tweet)
            st.markdown(f"**Text:** `{text_pred.upper()}` ({text_conf:.2f}) → {label_confidence(text_conf)}")

            # Image prediction
            if image_file:
                image = Image.open(image_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                clip_pred, clip_conf = predict_clip(image, tweet)
                st.markdown(f"**Image:** `{clip_pred.upper()}` (Similarity: {clip_conf:.2f}) → {label_confidence(clip_conf)}")
            else:
                clip_pred, clip_conf = "real", 0.5  # default neutral

            # Graph prediction
            if graph_file:
                try:
                    graph_data = torch.load(graph_file, map_location=device)
                    gnn_pred = predict_with_gnn(graph_data)
                    st.markdown(f"**Graph (GNN):** `{gnn_pred.upper()}`")
                    st.markdown("**Graph structure:**")
                    visualize_graph(graph_data)
                    visualize_tsne(load_gnn_model(), graph_data)
                except Exception as e:
                    st.error(f"GNN graph loading failed: {e}")
                    gnn_pred = "real"
            else:
                gnn_pred = "real"

            # Ensemble
            final_pred, final_score = ensemble_with_gnn(text_pred, text_conf, clip_pred, clip_conf, gnn_pred)
            st.markdown("---")
            if final_pred == "real":
                st.success(f"**Final Decision: REAL** (Score: {final_score}, {label_confidence(final_score)})")
            else:
                st.error(f"**Final Decision: FAKE** (Score: {final_score}, {label_confidence(final_score)})")
