import streamlit as st

from backend import EmailRequest, generate_email

st.set_page_config(
    page_title="AI Email Generator",
    page_icon="✉️",
    layout="centered",
)

st.title("✉️ AI Email Generator")
st.caption(
    "Create professional emails using OpenAI or Google Gemini. "
    "Enter your API key in the sidebar."
)

# ----------------------------------
# Sidebar: Provider and API key
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
        help="Your API key is used only for this session.",
    )

    if provider == "OpenAI":
        model = st.text_input(
            "OpenAI model",
            value="gpt-4o-mini",
        )
    else:
        model = st.text_input(
            "Gemini model",
            value="Gemini 3.8 Flash",
        )

    st.divider()
    st.info(
        "Do not share your API key publicly. "
        "For production apps, use Streamlit Secrets or environment variables."
    )

# ----------------------------------
# Main form: User inputs
# ----------------------------------

st.header("Email details")

with st.form("email_form"):
    purpose = st.text_area(
        "Purpose of email",
        placeholder=(
            "Example: Request a one-week extension for submitting the project "
            "report because additional time is needed for data analysis."
        ),
        height=140,
    )

    col1, col2 = st.columns(2)

    with col1:
        tone = st.selectbox(
            "Tone",
            options=[
                "Professional",
                "Formal",
                "Friendly",
                "Polite",
                "Assertive",
                "Persuasive",
                "Apologetic",
                "Enthusiastic",
            ],
        )

    with col2:
        recipient_type = st.selectbox(
            "Recipient type",
            options=[
                "Manager",
                "Client",
                "Professor",
                "HR / Recruiter",
                "Colleague",
                "Vendor",
                "Customer",
                "Team member",
            ],
        )

    recipient_name = st.text_input(
        "Recipient name (optional)",
        placeholder="Example: Priya Sharma",
    )

    sender_name = st.text_input(
        "Your name (optional)",
        placeholder="Example: Chaitra Sangal",
    )

    additional_details = st.text_area(
        "Additional details (optional)",
        placeholder=(
            "Add deadlines, project names, meeting links, attachments, "
            "or any other important information."
        ),
        height=120,
    )

    submitted = st.form_submit_button(
        "Generate email",
        use_container_width=True,
        type="primary",
    )

# ----------------------------------
# Generate and display output
# ----------------------------------

if submitted:
    if not purpose.strip():
        st.error("Please enter the purpose of the email.")
        st.stop()

    if not api_key.strip():
        st.error(f"Please enter your {provider} API key in the sidebar.")
        st.stop()

    request = EmailRequest(
        purpose=purpose.strip(),
        tone=tone,
        recipient_type=recipient_type,
        recipient_name=recipient_name.strip(),
        sender_name=sender_name.strip(),
        additional_details=additional_details.strip(),
    )

    with st.spinner("Generating your email..."):
        try:
            result = generate_email(
                provider=provider,
                api_key=api_key.strip(),
                model=model.strip(),
                request=request,
            )

            subject = result["subject"]
            body = result["body"]

            st.success("Email generated successfully!")

            st.subheader("Generated email")

            st.markdown(f"**Subject:** {subject}")
            st.markdown(body)

            st.divider()

            st.subheader("Editable email")
            st.text_area(
                "Copy or edit the generated email",
                value=f"Subject: {subject}\n\n{body}",
                height=350,
            )

            st.download_button(
                label="Download email",
                data=f"Subject: {subject}\n\n{body}",
                file_name="generated_email.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as error:
            st.error(
                "Could not generate the email. Please check your API key, "
                "selected model, and internet connection."
            )
            st.exception(error)