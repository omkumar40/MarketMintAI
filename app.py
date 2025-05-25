import streamlit as st
st.set_page_config(layout="wide", page_title="AI Marketing Content Generator PoC")

from PIL import Image
import io
import base64
import mimetypes  # To guess file extension from mime type
import uuid  # To generate unique filenames
import os
from google import genai
from google.genai import types

# --- Configuration ---
# Get API key from Streamlit secrets
# Initialize API key in session state if it doesn't exist
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state["GEMINI_API_KEY"] = None

# API Key Input
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state["GEMINI_API_KEY"])

# Update session state if the API key changes
if api_key and api_key != st.session_state["GEMINI_API_KEY"]:
    st.session_state["GEMINI_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

# --- Streamlit UI ---
st.title("AI Marketing Content Generator (PoC)")

# Only proceed if API key is provided
if not st.session_state["GEMINI_API_KEY"]:
    st.info("Please enter your Gemini API Key in the sidebar to start.")
    st.stop()

try:
    os.environ["GEMINI_API_KEY"] = st.session_state["GEMINI_API_KEY"]

except KeyError:
    st.error(
        "Gemini API Key not found in Streamlit secrets. Please add GEMINI_API_KEY='YOUR_API_KEY' to .streamlit/secrets.toml"
    )
    st.stop()
except Exception as e:
    st.error(
        f"Failed to configure Google AI models. Check your API key and environment. Error: {e}"
    )
    st.stop()

# Initialize models
try:
    # Use gemini-pro-vision for multimodal input (image + text).
    # NOTE: This model is for multimodal UNDERSTANDING and TEXT OUTPUT.
    # It is NOT designed for generating new images with embedded objects via standard SDK calls.
    # Attempting to get image output here is for demonstration of the requested call structure,
    # but is NOT expected to produce the required results.
    model_vision = "gemini-2.0-flash-preview-image-generation"
except Exception as e:
    st.error(
        f"Failed to initialize Google AI models. Ensure your API key is correct and the models are available. Error: {e}"
    )
    st.stop()





def display_ai_response(image_data_buffer, file_extension, task_name):
    """
    Processes the streaming AI response, looking for image data (inline_data)
    or text, and displays it in Streamlit.
    """
    st.markdown(f"**AI Response for {task_name}:**")

    if image_data_buffer:
        # Display the image if collected
        st.image(
            image_data_buffer,
            caption=f"{task_name} (AI Generated Image Attempt)",
            use_container_width=True,
        )
        # Optional: Provide a download link for the generated image attempt
        file_name = (
            f"{task_name.lower().replace(' ', '_').replace('-', '_')}_{uuid.uuid4()}{file_extension}"  # Added replace '-' for filename safety
        )
        st.download_button(
            label=f"Download {task_name} Image",
            data=image_data_buffer,
            file_name=file_name,
            mime=f"image/{file_extension[1:]}",
        )
    else:
        st.warning(f"AI model returned no image data for {task_name}.")


def generate_image(prompt, image_base64=None):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.0-flash-preview-image-generation"
    contents = []
    image_base64 = st.session_state["image_base64"]

    if image_base64:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        data=image_base64,
                        mime_type="image/jpeg",  # Adjust mime type if needed
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        )
    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            return data_buffer, file_extension
        else:
            st.write(chunk.text)
            return None, None


# --- Input Section ---
st.sidebar.header("Inputs")

uploaded_file = st.sidebar.file_uploader(
    "1. Upload Product Image (PNG, JPG)", type=["png", "jpg", "jpeg"]
)


if uploaded_file:
    if "image_base64" not in st.session_state:
        st.session_state["image_base64"] = base64.b64encode(uploaded_file.read()).decode("utf-8")
    image_base64 = st.session_state["image_base64"]

    st.sidebar.image(
        uploaded_file, caption="Uploaded Product", use_container_width=True
    )

    st.sidebar.subheader("2. Background Description (for Banner)")
    background_description = st.sidebar.text_area(
        "Describe the desired background scene for the product banner:",
        "Kids Watch displayed on a wooden study desk in a child’s bedroom",
    )

    st.sidebar.subheader("3. Human Model Description (for Model Images)")
    model_gender = st.sidebar.selectbox("Gender", ["Boy", "Girl", "Child (Gender Neutral)"])
    model_age = st.sidebar.selectbox("Age", ["Toddler", "Young Child (5-8)", "Older Child (9-12)"])
    model_ethnicity = st.sidebar.text_input(
        "Ethnicity (optional):", "Asian"
    )
    engagement_instruction_1 = st.sidebar.text_area(
        "Engagement for Pose 1 (e.g., wearing the watch, holding the toy, playing with it):",
        "A smiling boy wearing the kids watch, raising his hand.",
    )
    engagement_instruction_2 = st.sidebar.text_area(
        "Engagement for Pose 2:",
        "A girl holding the toy, sitting on the floor, looking at it curiously.",
    )

    st.sidebar.subheader("4. Ad Text & Tone (for Ad Creatives)")
    ad_header = st.sidebar.text_input("Header:", "Fun that Teaches Time!")
    ad_tagline = st.sidebar.text_input("Tagline:", "Durable, Educational, and Exciting!")
    ad_cta = st.sidebar.text_input("Call to Action (CTA):", "Shop Now!")
    ad_tone = st.sidebar.selectbox("Preferred Tone:", ["playful", "educational", "premium"])

    st.sidebar.markdown("---")
    generate_button = st.sidebar.button("Generate Marketing Content")

# --- Output Section ---
st.header("Generated Content")

if uploaded_file is None:
    st.info("Upload a product image in the sidebar to start generating marketing content.")
elif generate_button:
    #st.empty()  # Clear previous results

    # --- Task 1: Product Banner ---
    st.subheader("1. Product Banner (Custom Background)")
    # This prompt describes what a hypothetical AI *would* generate
    banner_prompt = f"""
Generate a photorealistic marketing banner image.
Scene description: {background_description}
Seamlessly integrate the provided product image into this scene.
Make the product a focal point on the desk.
Maintain product proportions and details.
The final image should be 1920x1080 pixels.
"""
    with st.spinner(f"Attempting AI Generation for Product Banner..."):
        try:
            # Attempt AI call for Task 1 (Background + Product Integration)
            # Note: Requesting IMAGE output modality
            image_base64 = st.session_state["image_base64"]
            image_data_buffer, file_extension = generate_image(
                banner_prompt, image_base64)
            # Display the response (which will likely be text or failure)
            display_ai_response(image_data_buffer, file_extension, "Product Banner")

        except Exception as e:
            st.error(f"API call failed for Product Banner: {e}")

    # --- Task 2: Human Model with Product ---
    st.subheader("2. Human Model with Product in Two Poses")

    # Construct the human model description correctly
    model_desc_parts = [model_age.lower(), model_gender.lower()]
    if model_ethnicity:
        model_desc_parts.append(model_ethnicity.lower())
    human_model_description_for_prompt = " ".join(model_desc_parts)

    # Pose 1
    prompt_pose1 = f"""
Generate a photorealistic marketing image of a human model interacting with the provided product.
Human model details: {human_model_description_for_prompt}
Interaction: {engagement_instruction_1}
Seamlessly integrate the provided product into the model's hand/wrist/body as described in the interaction.
Make the product clearly visible.
Output as a high-quality marketing image suitable for advertisements.
The final image should be 1080x1080 pixels.
"""
    with st.spinner(f"Attempting AI Generation for Human Model Image - Pose 1..."):
        try:
            print(prompt_pose1)
            # Attempt AI call for Task 2 Pose 1 (Human + Product Integration)
            image_base64 = st.session_state["image_base64"]
            image_data_buffer, file_extension = generate_image(
                prompt_pose1, image_base64)
            display_ai_response(image_data_buffer, file_extension, "Model Image - Pose 1")
        except Exception as e:
            st.error(f"API call failed for Model Image - Pose 1: {e}")

        # Pose 2
        prompt_pose2 = f"""
Generate a photorealistic marketing image of a human model interacting with the provided product.
Human model details: {human_model_description_for_prompt}
Interaction: {engagement_instruction_2}
Seamlessly integrate the provided product into the model's hand/lap/body as described in the interaction.
Make the product clearly visible.
Output as a high-quality marketing image suitable for advertisements.
The final image should be 1080x1080 pixels.
"""
        with st.spinner(f"Attempting AI Generation for Human Model Image - Pose 2..."):
            try:
                print(prompt_pose2)

                # Attempt AI call for Task 2 Pose 2 (Human + Product Integration)
                image_base64 = st.session_state["image_base64"]
                image_data_buffer, file_extension = generate_image(
                    prompt_pose2, image_base64)
                display_ai_response(image_data_buffer, file_extension, "Model Image - Pose 2")
            except Exception as e:
                st.error(f"API call failed for Model Image - Pose 2: {e}")

    # --- Task 3: Ad Creatives ---
    st.subheader("3. Ad Creatives")
    

    # Define platform specifics as provided
    platform_specifics = {
        "instagram": {
            "size": (1080, 1080),
            "description": "Square format perfect for Instagram feed.",
        },
        "facebook": {
            "size": (1200, 630),
            "description": "Horizontal layout optimized for Facebook feed.",
        },
        "website": {
            "size": (1920, 600),
            "description": "Wide banner optimized for website headers.",
        },
    }

    with st.spinner("Attempting AI Generation for Ad Creatives..."):
        for platform, details in platform_specifics.items():
            ad_size = details["size"]
            ad_description = details["description"]

            # Construct the AI prompt for Ad Creative
            # Note: Embedding text like Header/Tagline/CTA in the prompt
            # asking the AI to render it within the image is unlikely to work reliably.
            ad_creative_prompt = f"""
Generate a professional marketing ad creative for {platform.capitalize()}.
Platform Requirements: {ad_description}
Include the provided product image prominently in the layout.
The design should reflect a {ad_tone} tone.
Image dimensions should be approximately {ad_size[0]}x{ad_size[1]} pixels.

Include the following text elements within the image as part of the design:
Header: "{ad_header}"
Tagline: "{ad_tagline}"
Call to Action: "{ad_cta}"

Create a complete, ready-to-use ad creative that looks professionally designed.
"""
            st.markdown(f"**Ad Creative for {platform.capitalize()} ({ad_size[0]}x{ad_size[1]})**")
            #st.markdown("**Hypothetical AI Prompt Sent:**")
            #st.text(ad_creative_prompt)

            try:
                image_base64 = st.session_state["image_base64"]
                # Attempt AI call for Ad Creative (Product + Text + Specific Layout/Size - highly challenging for AI)
                image_data_buffer, file_extension = generate_image(
                    ad_creative_prompt, image_base64)
                display_ai_response(
                    image_data_buffer, file_extension, f"{platform.capitalize()} Ad Creative"
                )
            except Exception as e:
                st.error(f"API call failed for {platform.capitalize()} Ad Creative: {e}")
                

    st.success("Content generation attempt complete!")

# --- Footer/Disclaimer ---
st.markdown(
    """
---
Copyright © 2024 Om Kumar. All rights reserved.
This is a proof of concept (PoC) application for generating marketing content using AI. The AI model used here is primarily for understanding and generating text descriptions, and may not produce perfect images with embedded text as requested. For production use, consider integrating with dedicated image editing tools or services.
"""
)