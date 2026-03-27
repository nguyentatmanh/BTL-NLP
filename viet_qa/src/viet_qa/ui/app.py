import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Viet QA Benchmark", page_icon="🇻🇳", layout="wide")

st.title("Vietnamese QA Benchmark: Extractive vs Generative")
st.markdown("Enter a Vietnamese context and a question below to compare the two QA paradigms.")

context = st.text_area(
    "Context (Ngữ cảnh)", 
    height=200, 
    value="Đại học Bách khoa Hà Nội (viết tắt là HUST) là một trong những trường đại học kỹ thuật hàng đầu tại Việt Nam, được thành lập năm 1956. Trường nằm ở trung tâm thủ đô Hà Nội."
)
question = st.text_input("Question (Câu hỏi)", value="Trường Đại học Bách khoa Hà Nội được thành lập năm nào?")

if st.button("Compare Models", type="primary", use_container_width=True):
    if not context.strip() or not question.strip():
        st.warning("Please enter both context and question.")
    else:
        with st.spinner("Running inference on both models..."):
            try:
                res = requests.post(f"{API_URL}/compare", json={"context": context, "question": question})
                if res.status_code == 200:
                    data = res.json()
                    st.success("Comparison complete!")
                    
                    st.divider()
                    col_ext, col_gen = st.columns(2)
                    
                    # Extractive QA Column
                    with col_ext:
                        st.subheader("🔍 Extractive QA")
                        ext = data.get('extractive', {})
                        st.info(f"**Answer:**\n\n{ext.get('answer', 'N/A')}")
                        
                        e1, e2 = st.columns(2)
                        e1.metric("Confidence", f"{ext.get('confidence', 0)*100:.1f}%")
                        e2.metric("Latency", f"{ext.get('latency_ms', 0):.2f} ms")
                        
                        st.markdown(f"**Evidence:** {ext.get('evidence', '')}")
                        
                    # Generative QA Column
                    with col_gen:
                        st.subheader("🤖 Generative QA")
                        gen = data.get('generative', {})
                        st.success(f"**Answer:**\n\n{gen.get('answer', 'N/A')}")
                        
                        g1, g2 = st.columns(2)
                        g1.metric("Confidence", f"{gen.get('confidence', 0)*100:.1f}%")
                        g2.metric("Latency", f"{gen.get('latency_ms', 0):.2f} ms")
                        
                        st.markdown(f"**Evidence:** {gen.get('evidence', '')}")
                    
                    # Debug Section
                    with st.expander("🛠️ Debug Information"):
                        st.json(data)
                        
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to the backend at {API_URL}. Please ensure the FastAPI server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
