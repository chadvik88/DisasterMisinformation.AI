import gradio as gr
import torch
from PIL import Image
from app import predict_stance, predict_clip, predict_with_gnn, ensemble_with_gnn, label_confidence

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_demo(tweet_text, image, graph_file):
    # Text prediction
    text_pred, text_conf = predict_stance(tweet_text)

    # Image prediction (if provided)
    if image is not None:
        clip_pred, clip_conf = predict_clip(image, tweet_text)
    else:
        clip_pred, clip_conf = "real", 0.5

    # GNN prediction (if provided)
    if graph_file is not None:
        try:
            graph_data = torch.load(graph_file.name, map_location=device)
            gnn_pred = predict_with_gnn(graph_data)
        except Exception:
            gnn_pred = "real"
    else:
        gnn_pred = "real"

    # Ensemble final result
    final_pred, final_score = ensemble_with_gnn(text_pred, text_conf, clip_pred, clip_conf, gnn_pred)

    # Format output as Markdown
    output = (
        f"**Text Prediction**: `{text_pred}` ({text_conf:.2f}) → {label_confidence(text_conf)}  \n"
        f"**Image Prediction**: `{clip_pred}` (Similarity: {clip_conf:.2f}) → {label_confidence(clip_conf)}  \n"
        f"**GNN Prediction**: `{gnn_pred}`  \n"
        f"---  \n"
        f"**Final Ensemble**: `{final_pred.upper()}` → {label_confidence(final_score)} (Score: {final_score})"
    )
    return output

iface = gr.Interface(
    fn=run_demo,
    inputs=[
        gr.Textbox(label="Tweet Text"),
        gr.Image(label="Disaster Image (optional)", type="pil"),
        gr.File(label="Graph Data (.pt, optional)", type="filepath")
    ],
    outputs=gr.Markdown(),
    title="Real-Time Disaster Misinformation Detector",
    description="Upload a tweet, optional disaster image, and optional graph data. "
                "The system uses DeBERTa (text), CLIP (image), and a GNN (graph) to classify the content."
)

if __name__ == "__main__":
    iface.launch()
