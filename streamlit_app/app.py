import streamlit as st
from utils.scorer import analyze_url

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Webpage Quality Analyzer",
    layout="wide"
)

# ------------------- Title -------------------
st.title("Webpage Quality Analyzer")
st.caption("Analyze a webpage’s content quality, readability, and discover similar pages.")

# ------------------- Input -------------------
st.markdown("### Enter a Webpage URL")
url_input = st.text_input("URL:", placeholder="https://example.com/article")

if url_input:
    with st.spinner("Analyzing webpage... please wait"):
        try:
            result = analyze_url(url_input)

            if 'error' in result:
                st.error(f"Error fetching URL: {result['error']}")
            else:
                st.success("Analysis complete!")

                # ------------------- Layout Columns -------------------
                col1, col2, col3 = st.columns([1.8, 1, 1])

                with col1:
                    st.markdown("### Content Summary")
                    st.write(f"**Title:** {result.get('title', 'N/A')}")
                    st.write(f"**URL:** [{result['url']}]({result['url']})")

                with col2:
                    st.metric("Word Count", result.get('word_count', 'N/A'))
                    st.metric("Readability (Flesch)", result.get('readability', 'N/A'))

                with col3:
                    quality = result.get('quality_label', 'N/A')
                    conf_val = result.get('_model_confidence_pct')
                    conf_str = f"{conf_val:.2f}" if isinstance(conf_val, (int, float)) else ("N/A" if conf_val is None else str(conf_val))
                    st.metric("Quality Label", quality)
                    st.metric("Model Confidence (%)", conf_str)

                # ------------------- Thin Content -------------------
                # note: scorer uses 250 words as 'thin' threshold by default
                if result.get('is_thin'):
                    st.warning("This content may be thin or low quality (less than ~250 words).")
                else:
                    st.info("This content has sufficient length for article-style analysis.")

                # ------------------- Similar Pages -------------------
                st.markdown("---")
                st.subheader("Top Similar Pages")

                similar = result.get("similar_to", [])
                if similar:
                    for s in similar:
                        sim_score = round(s.get('similarity', 0.0) * 100, 2)
                        url = s.get('url', '')
                        st.markdown(f"[{url}]({url}) — **Similarity: {sim_score}%**")
                else:
                    st.write("No similar pages found.")

        except Exception as e:
            st.error(f"Error analyzing URL: {e}")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "Built using Streamlit | © 2025 Webpage Quality Analyzer"
    "</div>",
    unsafe_allow_html=True
)
