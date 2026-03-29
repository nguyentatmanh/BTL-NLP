import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Vietnamese QA System", page_icon="🇻🇳", layout="wide")

# Custom CSS to match the dark theme and layout from the screenshot
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #121212;
    }
    /* Section headers */
    h1, h4, h5 {
        color: #ffffff;
    }
    /* Secondary text */
    p, label {
        color: #b0b0b0;
    }
    /* Input fields (Text boxes) */
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #1e1e24 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
    }
    /* Red 'Trả lời' Button */
    .stButton > button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton > button:hover {
        background-color: #ff3333 !important;
    }
    /* Candidate cards in sidebar */
    .candidate-card {
        background-color: #1a1a24;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #ff4b4b;
    }
    .candidate-card p {
        margin-bottom: 0.2rem;
        font-size: 0.9rem;
    }
    /* Best Context Box */
    .best-context-box {
        background-color: #1e1e24;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #333333;
        color: #ffffff;
    }
    /* Highlight color inside context */
    .highlight {
        color: #ffb86c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("VN Hệ Thống Hỏi Đáp Tự Động Tiếng Việt")
st.markdown("*Hệ thống sử dụng TF-IDF (Retriever) và Extractive QA Transformer (Reader).*")

st.divider()

# Handle example click
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

# --- Input ---
question = st.text_input(
    "**Câu hỏi của bạn**",
    value=st.session_state["question_input"],
    placeholder="ai là hiệu trưởng đầu tiên của đại học bách khoa hà nội",
)

if st.button("🔍 Trả lời", use_container_width=True):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang tìm kiếm và trích xuất câu trả lời..."):
            try:
                res = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": 3},
                    timeout=60,
                )
                if res.status_code == 200:
                    data = res.json()

                    # --- Layout: Left (answer) | Right (candidates) ---
                    col_left, col_space, col_right = st.columns([5, 0.5, 4])

                    with col_left:
                        # Answer
                        st.markdown("#### Câu trả lời (Answer)")
                        st.text_input(
                            "answer_display", value=data["answer"],
                            disabled=True, label_visibility="collapsed",
                        )
                        st.write("") # Spacer

                        # Score + Status
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.markdown("**Final Score (0.0 - 1.0)**")
                            st.text_input(
                                "score_display", value=f"{data['final_score']:.3f}".replace('.', ','),
                                disabled=True, label_visibility="collapsed",
                            )
                        with sc2:
                            st.markdown("**Lý do / Trạng thái**")
                            st.text_input(
                                "status_display", value=data["status"],
                                disabled=True, label_visibility="collapsed",
                            )
                        
                        st.write("") # Spacer

                        # Best Context
                        st.markdown("#### Best Context (Ngữ cảnh)")
                        best_ctx = data.get("best_context", "")
                        answer_text = data["answer"]
                        
                        if answer_text and answer_text != "Không tìm thấy" and answer_text != "Error":
                            # Safe replacement ignoring casing
                            import re
                            pattern = re.compile(re.escape(answer_text), re.IGNORECASE)
                            highlighted = pattern.sub(f'<span class="highlight">{answer_text}</span>', best_ctx)
                            
                            st.markdown(
                                f'<div class="best-context-box">{highlighted}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="best-context-box">{best_ctx}</div>',
                                unsafe_allow_html=True,
                            )

                    with col_right:
                        st.markdown("#### Top Candidates (Debug)")
                        candidates = data.get("candidates", [])
                        for c in candidates:
                            st.markdown(
                                f'<div class="candidate-card">'
                                f'<p style="color:#ffffff; font-weight:bold; margin-bottom:0.5rem; font-size:1rem;">Top {c["rank"]}: {c["answer"]}</p>'
                                f'<p>- Final Score: {c["final_score"]:.3f}</p>'
                                f'<p>- Reader Score: {c["reader_score"]:.3f}</p>'
                                f'<p>- Retriever Score: {c["retriever_score"]:.3f}</p>'
                                f'<p>- Context (snippet): {c["context_snippet"][:150]}...</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                elif res.status_code == 503:
                    st.warning("⏳ Retriever chưa sẵn sàng. Vui lòng đợi hệ thống load xong dataset.")
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Không thể kết nối đến backend ({API_URL}). "
                    "Hãy chạy `uvicorn src.viet_qa.api.main:app --reload --port 8000` trước."
                )
            except Exception as e:
                st.error(f"Lỗi: {e}")

# --- Example questions ---
st.write("")
st.write("")
st.markdown("##### 📋 Examples")

examples = [
    "Quốc kỳ Vương quốc Liên hiệp Anh và Bắc Ireland trường được...",
    "Hà Nội nằm ở đâu?",
    "Dân số Việt Nam năm 2024 là bao nhiêu?",
    "Đại dịch COVID-19 bắt nguồn từ đâu?",
]

cols = st.columns(4)
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["question_input"] = ex
            st.rerun()
