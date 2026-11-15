import re
from collections import Counter
from io import BytesIO

import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader

# ----------------------------------
# Page configuration
# ----------------------------------

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Text Summarizer")
st.caption(
    "Paste your text below and let AI summarize it. "
    "You can also upload a PDF, Word document, text file, or Excel file."
)

# ----------------------------------
# Sidebar: AI provider and API key
# ----------------------------------

with st.sidebar:
    st.header("AI Provider")

    provider = st.selectbox(
        "Select AI provider",
        options=["OpenAI", "Google Gemini"],
        index=0,
    )

    api_key = st.text_input(
        f"{provider} API key",
        type="password",
        help="Your API key is used only for this browser session.",
    )

    if provider == "OpenAI":
        model = st.text_input(
            "OpenAI model",
            value="gpt-4o-mini",
        )
    else:
        model = st.text_input(
            "Gemini model",
            value="gemini-2.0-flash",
        )

    st.divider()

    st.info(
        "For production applications, use Streamlit Secrets or environment "
        "variables instead of entering API keys in the browser."
    )

# ----------------------------------
# Helper functions
# ----------------------------------

SUMMARY_INSTRUCTIONS = {
    "Short": (
        "Write a concise summary in 3 to 5 sentences. "
        "Include only the most important points."
    ),
    "Medium": (
        "Write a clear and balanced summary in 1 to 2 paragraphs. "
        "Include the main ideas and important supporting points."
    ),
    "Detailed": (
        "Write a detailed summary in 3 to 5 paragraphs. "
        "Include the main ideas, supporting points, conclusions, "
        "and important details."
    ),
}


def clean_text(text: str) -> str:
    """Clean extracted text and preserve paragraph spacing."""

    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF file."""

    pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []

    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    return "\n\n".join(pages)


def extract_text_from_docx(uploaded_file) -> str:
    """Extract text from a Word document."""

    document = Document(BytesIO(uploaded_file.getvalue()))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]

    return "\n".join(paragraphs)


def extract_text_from_excel(uploaded_file) -> str:
    """Extract readable text from all sheets in an Excel file."""

    excel_file = pd.ExcelFile(BytesIO(uploaded_file.getvalue()))
    parts = []

    for sheet_name in excel_file.sheet_names:
        dataframe = excel_file.parse(sheet_name, header=None)
        dataframe = dataframe.fillna("")

        parts.append(f"Sheet: {sheet_name}")
        parts.append(dataframe.to_string(index=False, header=False))

    return "\n\n".join(parts)


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from PDF, DOCX, TXT, XLSX, or XLS files."""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(uploaded_file)

    elif file_name.endswith(".docx"):
        extracted_text = extract_text_from_docx(uploaded_file)

    elif file_name.endswith(".txt"):
        extracted_text = uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore",
        )

    elif file_name.endswith((".xlsx", ".xls")):
        extracted_text = extract_text_from_excel(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF, DOCX, TXT, "
            "XLSX, or XLS file."
        )

    return clean_text(extracted_text)


def build_summary_prompt(text: str, summary_length: str) -> str:
    """Create a dynamic prompt based on the selected summary length."""

    instruction = SUMMARY_INSTRUCTIONS.get(
        summary_length,
        SUMMARY_INSTRUCTIONS["Medium"],
    )

    return f"""You are an expert text-summarization assistant.

Summarize the following text.

SUMMARY LENGTH
{summary_length}

INSTRUCTIONS
{instruction}

RULES
1. Capture the main ideas accurately.
2. Do not add information that is not present in the original text.
3. Use clear, simple, professional language.
4. Preserve important facts, figures, names, and conclusions where relevant.
5. Do not include phrases such as "Here is the summary" or "In conclusion".

TEXT TO SUMMARIZE:
\"\"\"
{text}
\"\"\"

Return only the summary text.
"""


def generate_summary_with_openai(
    api_key: str,
    model: str,
    text: str,
    summary_length: str,
) -> str:
    """Generate a summary using the OpenAI API."""

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert text summarizer. "
                    "Write accurate, clear, and concise summaries."
                ),
            },
            {
                "role": "user",
                "content": build_summary_prompt(text, summary_length),
            },
        ],
        temperature=0.3,
    )

    return (response.choices[0].message.content or "").strip()


def generate_summary_with_gemini(
    api_key: str,
    model: str,
    text: str,
    summary_length: str,
) -> str:
    """Generate a summary using the Google Gemini API."""

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=build_summary_prompt(text, summary_length),
        config=types.GenerateContentConfig(
            temperature=0.3,
        ),
    )

    return (response.text or "").strip()


def generate_summary(
    provider: str,
    api_key: str,
    model: str,
    text: str,
    summary_length: str,
) -> str:
    """Call the selected AI provider and return the summary."""

    if provider == "OpenAI":
        summary = generate_summary_with_openai(
            api_key=api_key,
            model=model,
            text=text,
            summary_length=summary_length,
        )

    elif provider == "Google Gemini":
        summary = generate_summary_with_gemini(
            api_key=api_key,
            model=model,
            text=text,
            summary_length=summary_length,
        )

    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

    if not summary:
        raise ValueError("The AI provider returned an empty summary.")

    return summary


def get_text_analytics(original_text: str, summary_text: str) -> dict:
    """Calculate analytics for the original text and generated summary."""

    original_text = original_text.strip()
    summary_text = summary_text.strip()

    original_words = re.findall(r"\b[\w']+\b", original_text.lower())
    summary_words = re.findall(r"\b[\w']+\b", summary_text.lower())

    original_word_count = len(original_words)
    summary_word_count = len(summary_words)

    compression_ratio = 0.0

    if original_word_count > 0:
        compression_ratio = (
            (original_word_count - summary_word_count)
            / original_word_count
            * 100
        )

    words_per_minute = 225

    original_reading_time_minutes = original_word_count / words_per_minute
    summary_reading_time_minutes = summary_word_count / words_per_minute

    reading_time_saved_minutes = max(
        0,
        original_reading_time_minutes - summary_reading_time_minutes,
    )

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "than",
        "of", "to", "in", "on", "for", "with", "as", "by", "at", "from",
        "is", "are", "was", "were", "be", "been", "being", "it", "its",
        "this", "that", "these", "those", "there", "here", "we", "you",
        "they", "he", "she", "his", "her", "their", "our", "your", "i",
        "not", "no", "yes", "can", "will", "would", "should", "could",
        "may", "might", "must", "do", "does", "did", "have", "has", "had",
    }

    meaningful_words = [
        word for word in original_words
        if word not in stop_words and len(word) > 2
    ]

    top_words = Counter(meaningful_words).most_common(10)

    return {
        "original_word_count": original_word_count,
        "summary_word_count": summary_word_count,
        "original_character_count": len(original_text),
        "summary_character_count": len(summary_text),
        "compression_ratio": compression_ratio,
        "original_reading_time_minutes": original_reading_time_minutes,
        "reading_time_saved_minutes": reading_time_saved_minutes,
        "top_words": top_words,
    }


# ----------------------------------
# Main input area
# ----------------------------------

st.header("Enter your text")
st.write("Paste your article, paragraph or notes here...")

input_method = st.radio(
    "Choose input method",
    options=["Paste text", "Upload file"],
    horizontal=True,
)

text_input = ""

if input_method == "Paste text":
    text_input = st.text_area(
        "Enter your text",
        height=280,
        placeholder=(
            "Example: Artificial intelligence is transforming industries by "
            "automating repetitive tasks, improving decision-making, and "
            "enabling new products and services..."
        ),
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "docx", "txt", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        st.success(f"Uploaded file: {uploaded_file.name}")

        try:
            with st.spinner("Extracting text from the file..."):
                text_input = extract_text_from_file(uploaded_file)

            if text_input.strip():
                st.success(
                    f"Text extracted successfully. "
                    f"Extracted characters: {len(text_input):,}"
                )
            else:
                st.warning(
                    "No readable text could be extracted from this file. "
                    "The file may be scanned, image-based, empty, or protected."
                )

        except Exception as error:
            st.error(f"Could not read the uploaded file: {error}")

# ----------------------------------
# Summary settings
# ----------------------------------

st.header("Summary settings")

col1, col2 = st.columns([2, 1])

with col1:
    summary_length = st.selectbox(
        "Choose summary length",
        options=["Short", "Medium", "Detailed"],
        index=1,
    )

with col2:
    st.write("")
    st.write("")
    generate_summary_button = st.button(
        "Generate Summary",
        type="primary",
        use_container_width=True,
    )

# ----------------------------------
# Generate and display summary
# ----------------------------------

if generate_summary_button:
    if not text_input or not text_input.strip():
        st.error("Please paste text or upload a readable file.")
        st.stop()

    if not api_key.strip():
        st.error(f"Please enter your {provider} API key in the sidebar.")
        st.stop()

    with st.spinner("Generating your AI summary..."):
        try:
            summary = generate_summary(
                provider=provider,
                api_key=api_key.strip(),
                model=model.strip(),
                text=text_input.strip(),
                summary_length=summary_length,
            )

            st.success("Summary generated successfully!")

            st.subheader("Generated Summary")
            st.markdown(summary)

            st.divider()

            st.subheader("Text analytics")

            analytics = get_text_analytics(
                original_text=text_input,
                summary_text=summary,
            )

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric(
                    label="Original words",
                    value=f"{analytics['original_word_count']:,}",
                )

            with col_b:
                st.metric(
                    label="Summary words",
                    value=f"{analytics['summary_word_count']:,}",
                )

            with col_c:
                st.metric(
                    label="Compression",
                    value=f"{analytics['compression_ratio']:.1f}%",
                )

            with col_d:
                st.metric(
                    label="Reading time saved",
                    value=f"{analytics['reading_time_saved_minutes']:.1f} min",
                )

            col_e, col_f, col_g = st.columns(3)

            with col_e:
                st.metric(
                    label="Original characters",
                    value=f"{analytics['original_character_count']:,}",
                )

            with col_f:
                st.metric(
                    label="Summary characters",
                    value=f"{analytics['summary_character_count']:,}",
                )

            with col_g:
                st.metric(
                    label="Original reading time",
                    value=f"{analytics['original_reading_time_minutes']:.1f} min",
                )

            st.subheader("Most frequent words")

            frequent_words = analytics["top_words"]

            if frequent_words:
                for word, count in frequent_words:
                    st.markdown(
                        f"- **{word}** — {count} occurrence"
                        f"{'s' if count != 1 else ''}"
                    )
            else:
                st.info(
                    "No meaningful words were found for word-frequency analysis."
                )

            st.download_button(
                label="Download summary",
                data=summary,
                file_name="ai_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as error:
            st.error(
                "Could not generate the summary. Please check your API key, "
                "selected model, and internet connection."
            )
            st.exception(error)

# ----------------------------------
# Placeholder before generation
# ----------------------------------

else:
    st.subheader("Generated Summary")
    st.info("Your AI-generated summary will appear here.")