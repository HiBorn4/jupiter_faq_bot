import streamlit as st
from faq_bot import answer
# from bing_image_downloader import downloader  # for suggestion icons (optional)
from faq_bot import retrieve  # for related questions

st.set_page_config(page_title="Jupiter FAQ Bot")

st.title("Jupiter FAQ Bot 🤖")

lang = st.sidebar.selectbox("Language", ["English", "Hindi", "Hinglish"], index=0)
query = st.text_input("Ask your question:")

if st.button("Send") and query:
    result = answer(query, lang=lang.lower())
    st.markdown(f"**Bot:** {result['answer']}")
    if result['source']:
        st.caption(f"(Based on: '{result['source']}')")
    # Suggest related queries
    related = [q for q,_,_ in retrieve(query, k=5)]
    with st.expander("Related questions:"):
        for rq in related:
            st.write(f"- {rq}")