import streamlit as st
import requests
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Vietnamese QA System", page_icon="🇻🇳", layout="wide")

# Custom CSS for Blue Theme
st.markdown("""
<style>
    .stApp { background-color: #121212; }
    h1, h2, h3, h4, h5 { color: #ffffff; }
    p, label { color: #b0b0b0; }
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #1a1a24 !important;
        color: #ffffff !important;
        border: 1px solid #2a2a3a !important;
        border-radius: 6px !important;
    }
    .stButton > button {
        background-color: #007BFF !important; /* Xanh dương */
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton > button:hover {
        background-color: #0056b3 !important;
    }
    .candidate-card {
        background-color: #1a1a24;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #007BFF; /* Xanh dương */
    }
    .candidate-card p {
        margin-bottom: 0.2rem;
        font-size: 0.9rem;
    }
    .best-context-box {
        background-color: #1a1a24;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #2a2a3a;
        color: #ffffff;
    }
    .highlight {
        color: #00aaff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("HỆ THỐNG HỎI ĐÁP TIẾNG VIỆT TỰ ĐỘNG")
st.markdown("*Hệ thống sử dụng kiến trúc Retriever-Reader với BM25 Retriever, XLM-RoBERTa Extractive Reader và Qwen2.5 Generative Reader.*")
st.markdown("*Hệ thống được xây dựng và đánh giá trên bộ dữ liệu tiếng Việt từ dataset `ntphuc149/ViSpanExtractQA`*")

st.divider()

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

# --- Khung Giao Diện Chính (Input Panel) ---
question = st.text_input(
    "**Câu hỏi của bạn**",
    value=st.session_state["question_input"],
    placeholder="ai là hiệu trưởng đầu tiên của đại học bách khoa hà nội",
)

reader_type = st.radio(
    "**🧠 Chọn Chế Độ Trả Lời (Reader Engine)**",
    options=["extractive", "generative", "both"],
    format_func=lambda x: "Chỉ chạy Extractive (Mô hình Train)" if x == "extractive" else ("Chỉ chạy Generative (Qwen1.5B)" if x == "generative" else "🚀 Chạy Song Song Cả 2 (So Sánh Tốc độ/Chất lượng)"),
    horizontal=True
)

def render_model_result(q, model_str, title, candidate_col=None):
    st.markdown(f"#### {title}")
    start_time = time.time()
    try:
        res = requests.post(
            f"{API_URL}/ask",
            json={"question": q, "top_k": 3, "model_type": model_str},
            timeout=180,
        )
        elapsed = time.time() - start_time
        if res.status_code == 200:
            data = res.json()
            st.markdown("**Câu trả lời (Answer)**")
            st.text_input(
                f"ans_{model_str}", value=data["answer"],
                disabled=True, label_visibility="collapsed",
            )
            
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Điểm tự tin (Final Score)**")
                st.text_input(
                    f"sc_{model_str}", value=f"{data['final_score']:.3f}".replace('.', ','),
                    disabled=True, label_visibility="collapsed",
                )
            with sc2:
                st.markdown("**Tốc độ phản hồi (Latency)**")
                # Highlight in green if fast
                time_color = "🟢" if elapsed < 1.0 else "🟡" if elapsed < 5.0 else "🔴"
                st.text_input(
                    f"tm_{model_str}", value=f"{time_color} {elapsed:.2f}s",
                    disabled=True, label_visibility="collapsed",
                )
            
            st.write("")
            st.markdown("**Best Context (Ngữ cảnh chốt)**")
            best_ctx = data.get("best_context", "")
            ans_text = data["answer"]
            
            if ans_text and ans_text != "Không tìm thấy" and ans_text != "Error":
                import re
                import html
                
                # HTML escape để tránh việc ngữ cảnh Wikipedia chứa dấu < > làm vỡ layout của trình duyệt
                safe_ctx = html.escape(best_ctx)
                safe_ans = html.escape(ans_text)
                
                pattern = re.compile(re.escape(safe_ans), re.IGNORECASE)
                highlighted = pattern.sub(f'<span class="highlight">{safe_ans}</span>', safe_ctx)
                
                # Để cho văn bản dễ đọc hơn, ta thay thế \n bằng thẻ <br>
                highlighted = highlighted.replace("\n", "<br>")
                
                st.markdown(
                    f'<div class="best-context-box">{highlighted}</div>',
                    unsafe_allow_html=True,
                )
            else:
                import html
                safe_ctx = html.escape(best_ctx).replace("\n", "<br>")
                st.markdown(
                    f'<div class="best-context-box">{safe_ctx}</div>',
                    unsafe_allow_html=True,
                )
                
            # --- In Bảng xếp hạng các ứng viên tiềm năng bị loại (Phục vụ debug) ---
            candidates = data.get("candidates", [])
            
            if candidate_col is None:
                # Chế độ Compare: Hiển thị bên dưới dạng Expander để tiết kiệm diện tích
                st.write("")
                with st.expander(f"Hiển thị Top 3 Ứng viên (Debug)"):
                    for c in candidates:
                        st.markdown(
                            f'<div class="candidate-card">'
                            f'<p style="color:#ffffff; font-weight:bold; margin-bottom:0.5rem; font-size:1rem;">Top {c["rank"]}: {c["answer"]}</p>'
                            f'<p>- Khớp toàn cục (Final): {c["final_score"]:.3f}</p>'
                            f'<p>- Lực đọc (Reader): {c["reader_score"]:.3f}</p>'
                            f'<p>- Căn chỉnh ngữ cảnh (Retriever): {c["retriever_score"]:.3f}</p>'
                            f'<p>- Trích đoạn: {c["context_snippet"][:150]}...</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                # Chế độ Single: Hiển thị sang hẳn cột bên phải
                with candidate_col:
                    st.markdown("#### Top Candidates (Debug)")
                    for c in candidates:
                        st.markdown(
                            f'<div class="candidate-card">'
                            f'<p style="color:#ffffff; font-weight:bold; margin-bottom:0.5rem; font-size:1rem;">Top {c["rank"]}: {c["answer"]}</p>'
                            f'<p>- Khớp toàn cục (Final): {c["final_score"]:.3f}</p>'
                            f'<p>- Lực đọc (Reader): {c["reader_score"]:.3f}</p>'
                            f'<p>- Căn chỉnh ngữ cảnh (Retriever): {c["retriever_score"]:.3f}</p>'
                            f'<p>- Trích đoạn: {c["context_snippet"][:150]}...</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    
        elif res.status_code == 503:
            st.warning("⏳ Retriever chưa sẵn sàng tải dữ liệu.")
        else:
            st.error(f"API Error ({res.status_code}): {res.text}")
            
    except requests.exceptions.ConnectionError:
        st.error(f"Mất kết nối API Backend ({API_URL}).")
    except Exception as e:
        st.error(f"Lỗi: {e}")

if st.button("🔍 Trả lời", use_container_width=True):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        st.write("---")
        with st.spinner("Hệ thống đang chạy phân tích..."):
            if reader_type in ["extractive", "generative"]:
                col_left, col_space, col_right = st.columns([5, 0.5, 4])
                title = "🧠 Mô Hình Extractive (Bám trụ văn bản)" if reader_type == "extractive" else "🤖 Mô Hình Generative (LLM Qwen2.5)"
                with col_left:
                    render_model_result(question, reader_type, title, candidate_col=col_right)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    render_model_result(question, "extractive", "🧠 Mô Hình Extractive")
                with c2:
                    render_model_result(question, "generative", "🤖 Mô Hình Generative")
