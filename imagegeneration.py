import boto3
import json
import base64
from io import BytesIO
import streamlit as st

# AWS Bedrock client setup for Titan Image Generator model
aws_bedrock = boto3.client('bedrock-runtime', region_name='ap-south-1')
bedrock_model_id = "amazon.titan-image-generator-v1"

def decode_image_from_response(response):
    """
    Decodes the image from the model's response.
    """
    try:
        # Read the StreamingBody response
        response_body = response['body'].read()

        # Decode the JSON response
        response_json = json.loads(response_body)

        # Extract the base64-encoded image data
        base64_image_data = response_json['images'][0]  # Assuming 'images' is the key for image data

        # Decode the base64 image data
        image_bytes = base64.b64decode(base64_image_data)

        return BytesIO(image_bytes)
    except Exception as e:
        st.error(f"Error decoding image: {e}")



def generate_image(prompt, style):
    """
    Generates an image based on the provided prompt and style.
    """
    try:
        request_payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": f"{style} {prompt}"},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "cfgScale": 9.0,
                "height": 512,
                "width": 512,
                "seed": 123  # You can use a random seed generation logic here if needed
            }
        }
        
        response = aws_bedrock.invoke_model(
            body=json.dumps(request_payload),
            modelId=bedrock_model_id
        )
        
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return decode_image_from_response(response)
        else:
            st.error(f"Error invoking model: {response.get('body')}")
            return None
    except aws_bedrock.exceptions.ValidationException as ve:
        st.error(f"ValidationException: {ve}")
    except Exception as e:
        st.error(f"Error generating image: {e}")

# Streamlit interface setup with custom styles
st.set_page_config(page_title="Personalized Artwork Creator", layout="wide")

st.title("Personalized image generation")
prompt_column, result_column = st.columns(2)

with prompt_column:
    st.subheader("Idea box")
    prompt_text = st.text_area("Write your prompt", height=150)
    art_style = st.selectbox("Select art style", ["Abstract", "Cute", "Fantasy", "Futuristic", "Realistic", "Science Fiction", "Surreal", "Techno"])
    generate_button = st.button("Generate Image")

with result_column:
    st.subheader("Generated Artwork")
    if generate_button:
        with st.spinner("Generating..."):
            artwork_image = generate_image(prompt_text, art_style)
            if artwork_image:
                st.image(artwork_image, caption='Generated Image', use_column_width=True)
