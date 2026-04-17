import os
import streamlit as st

from src.explanations.advisor_explainability import (
    generate_rag_explanation,
    get_highlight_terms,
    get_matched_terms,
    highlight_html,
    render_term_pills,
)
from src.search_engines.chroma_index import initialize_chroma_database
from src.search_engines.chroma_engine import ChromaSearchEngine
from src.search_engines.llm_search import llm_search_advisors
from src.advisors.match_output import MatchAdvisor
from config import ENABLE_LLM_SEARCH

try:
    from src.generators.advisor_profile_enricher import fetch_and_update_advisors
except ImportError:
    fetch_and_update_advisors = None

st.set_page_config(
    page_title="Oracle Advisor Search",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource
def get_search_engine() -> ChromaSearchEngine:
    return initialize_chroma_database()


def render_result_card(
    query: str, result: MatchAdvisor, explanation: str | None = None
) -> None:
    advisor = result.advisor
    score = result.score
    document = result.document
    matched_terms = get_matched_terms(query, advisor)
    highlight_terms = get_highlight_terms(query, advisor)

    with st.container(border=True):
        left_col, right_col = st.columns([4, 1])
        with left_col:
            st.subheader(advisor.name)
            st.caption(f"{advisor.title} · {advisor.section}")
        with right_col:
            st.metric("Match", f"{score:.2f}")

        st.write(f"**Email:** {advisor.email}")

        if matched_terms:
            st.write("**Matched terms**")
            st.write(", ".join(matched_terms))

        if highlight_terms:
            st.write("**Highlighted match words**")
            st.markdown(render_term_pills(highlight_terms), unsafe_allow_html=True)

        if explanation:
            st.write("**Why this advisor matches**")
            st.markdown(
                highlight_html(explanation, highlight_terms), unsafe_allow_html=True
            )

        if document:
            st.write("**Evidence from advisor profile**")
            preview = document[:600]
            if len(document) > 600:
                preview += "..."
            st.markdown(
                highlight_html(preview, highlight_terms), unsafe_allow_html=True
            )

        with st.expander("Profile details"):
            st.write("**Research output**")
            st.markdown(
                highlight_html(
                    "; ".join(advisor.research_output) or "N/A", highlight_terms
                ),
                unsafe_allow_html=True,
            )
            st.write("**Activities**")
            st.markdown(
                highlight_html("; ".join(advisor.activities) or "N/A", highlight_terms),
                unsafe_allow_html=True,
            )
            st.write("**Press/Media**")
            st.markdown(
                highlight_html(
                    "; ".join(advisor.press_media) or "N/A", highlight_terms
                ),
                unsafe_allow_html=True,
            )


def main() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero {
                padding: 1.25rem 1.5rem;
                border-radius: 1.25rem;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 12px 40px rgba(15, 23, 42, 0.22);
            }
            .hero h1 {
                margin: 0;
                font-size: 2rem;
            }
            .hero p {
                margin: 0.4rem 0 0;
                color: rgba(255, 255, 255, 0.85);
                font-size: 1rem;
            }
            .match-pill {
                display: inline-block;
                margin: 0.15rem 0.35rem 0.15rem 0;
                padding: 0.18rem 0.6rem;
                border-radius: 999px;
                background: #e0f2fe;
                color: #075985;
                border: 1px solid #7dd3fc;
                font-size: 0.82rem;
                font-weight: 600;
            }
            mark.highlight-term {
                background: #fde68a;
                color: inherit;
                padding: 0.05rem 0.2rem;
                border-radius: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Oracle Advisor Search</h1>
            <p>Describe your thesis interests in the chat box, then search for matching advisors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([3, 1], gap="large")

    with left_col:
        st.subheader("Chat")
        st.write(
            "Ask for advisors by name, research topic, publications, department, or email."
        )

        query = st.text_area(
            "Your message",
            placeholder="Example: I want an advisor for energy systems.",
            height=140,
            key="query_input",
            label_visibility="collapsed",
        )

        search_clicked = st.button("Search advisors", type="primary")

        if search_clicked:
            if not query.strip():
                st.warning("Enter a query before searching.")
            else:
                with st.spinner("Searching advisors..."):
                    engine = get_search_engine()
                    api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
                    llm_error = None

                    if ENABLE_LLM_SEARCH:
                        if not api_key:
                            st.warning(
                                "LLM search is enabled in config, but no OpenRouter API key is provided."
                            )
                            results = []
                        else:
                            results, llm_error = llm_search_advisors(
                                query=query.strip(),
                                advisors=list(engine.get_all_advisors().values()),
                                top_k=5,
                                api_key=api_key,
                            )
                    else:
                        results = engine.search(query.strip(), top_k=5)

                    if not results:
                        if ENABLE_LLM_SEARCH and llm_error:
                            st.warning(f"LLM search issue: {llm_error}")
                        st.info("No strong matches were found for this query.")
                    else:
                        result_dicts = []
                        for match in results:
                            result_dicts.append(
                                {
                                    "advisor": match.advisor,
                                    "score": match.score,
                                    "document": match.document,
                                }
                            )
                        explanation = generate_rag_explanation(
                            query.strip(),
                            result_dicts,
                            api_key=api_key or None,
                        )

                        st.session_state["last_query"] = query.strip()
                        st.session_state["last_results"] = results
                        st.session_state["last_explanation"] = explanation

        if st.session_state.get("last_results"):
            st.subheader("Results")
            explanation_text = st.session_state.get("last_explanation")
            for result in st.session_state["last_results"]:
                render_result_card(
                    st.session_state.get("last_query", ""), result, explanation_text
                )

    with right_col:
        st.subheader("List of advisors in database")
        engine = get_search_engine()
        stats = engine.get_collection_stats()
        st.write(f"Collection: {stats['collection_name']}")
        st.write(f"Stored advisors: {stats['total_advisors']}")

        st.subheader("Fetch new advisor data")
        if st.button("Fetch and update advisors", type="secondary"):
            if fetch_and_update_advisors is None:
                st.info(
                    "Advisor refresh is not available in this checkout because generators/advisor_profile_enricher.py is missing."
                )
            else:
                with st.spinner("Fetching advisor data and updating database..."):
                    fetch_and_update_advisors()
                    engine = get_search_engine()
                    stats = engine.get_collection_stats()
                    st.success("Advisor data updated successfully!")

        
        if ENABLE_LLM_SEARCH:
            st.subheader("API Key")
            st.write(
                "Required when LLM search is enabled in config. Also used for optional explanations."
            )
        
            st.text_input(
                "OpenRouter API key",
                type="password",
                placeholder="Paste your API key here",
                key="api_key",
                help="Used only for explanation text generation after the search finishes.",
            )

            st.divider()
            st.caption("The key stays in your Streamlit session while the app is running.")


if __name__ == "__main__":
    main()
