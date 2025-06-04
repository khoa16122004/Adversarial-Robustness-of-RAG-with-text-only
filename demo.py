import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils import get_font, DataLoader
from reader import Reader

def format_text_with_br(text, max_line_length=80):
    if not isinstance(text, str):
        return "None"
    lines = []
    current_line_start = 0
    while current_line_start < len(text):
        cut_at = min(current_line_start + max_line_length, len(text))
        if cut_at < len(text):
            last_space = text.rfind(' ', current_line_start, cut_at)
            if last_space > current_line_start:
                cut_at = last_space + 1
        lines.append(text[current_line_start:cut_at])
        current_line_start = cut_at
    return "<br>".join(lines)

# === Load dữ liệu ===
dir_ = r"llama_7b_nsgaii_logs"
dataset = DataLoader("data_new_v2.json")
model_name = "llama-7b"
sample_id = 0

merge_font_data = get_font(dir_, model_name, sample_id)
reader = Reader(model_name)
print("Done init reader")
original_document, question, gt_answer, answer_position_indices = dataset.take_sample(sample_id)

retriever_scores = merge_font_data[:, 0].astype(float)
reader_scores = merge_font_data[:, 1].astype(float)
perturbed_texts_raw = [ind[2].get_perturbed_text() for ind in merge_font_data]

# === Khởi tạo session state nếu chưa có ===
if "outputs_raw" not in st.session_state:
    st.session_state.outputs_raw = ["None"] * len(perturbed_texts_raw)

# === Tạo các danh sách hiển thị ===
indexes = list(range(len(perturbed_texts_raw)))
perturbed_texts_for_display = [format_text_with_br(text) for text in perturbed_texts_raw]
outputs_for_display = [format_text_with_br(output) for output in st.session_state.outputs_raw]

# === UI Streamlit ===
st.set_page_config(layout="wide")
st.title("📊 Document Analysis Interface")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 🧠 Ground Truth Information")
    st.write("**Question:**", question)
    st.write("**GT Answer:**", gt_answer)
    st.write("**Original Document:**", original_document)

    if st.button("▶️ Run Inference"):
        try:
            # chạy inference cho toàn bộ perturbed_texts_raw một lần
            outputs = reader.generate(question, perturbed_texts_raw)
            # đảm bảo outputs có độ dài đúng
            if not isinstance(outputs, list) or len(outputs) != len(perturbed_texts_raw):
                outputs = ["Invalid Output"] * len(perturbed_texts_raw)
        except Exception as e:
            outputs = [f"Error: {e}"] * len(perturbed_texts_raw)

        st.session_state.outputs_raw = outputs
        st.success("Inference complete! Please hover again to see updated output.")


with col2:
    st.markdown("### 📉 Pareto Front")

    # Cập nhật dữ liệu hiển thị
    outputs_for_display = [format_text_with_br(output) for output in st.session_state.outputs_raw]
    perturbed_texts_for_display = [format_text_with_br(text) for text in perturbed_texts_raw]
    customdata = list(zip(indexes, perturbed_texts_for_display, outputs_for_display))

    fig = go.Figure(data=go.Scatter(
        x=retriever_scores,
        y=reader_scores,
        mode='markers',
        marker=dict(
            size=10,
            color='deepskyblue',
            line=dict(width=1, color='DarkSlateGrey')
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>Index:</b> %{customdata[0]}<br><br>"
            "<b>Perturbed Text:</b><br>%{customdata[1]}<br><br>"
            "<b>Output:</b><br>%{customdata[2]}<extra></extra>"
        )
    ))

    fig.update_layout(
        xaxis_title="Retriever Score",
        yaxis_title="Reader Score",
        template="plotly_dark",
        font=dict(family="Arial, sans-serif", size=14, color="white"),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="rgba(30, 30, 30, 0.9)",
            bordercolor="rgba(150, 150, 150, 0.7)",
            font=dict(size=13, color="white"),
            align="left",
            namelength=-1
        ),
        margin=dict(l=60, r=60, t=70, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)
