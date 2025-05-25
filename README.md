# MarketMintAI: AI Marketing Content Generator (PoC)

MarketMintAI is a proof-of-concept Streamlit web application that leverages Google Gemini AI to help marketers and product owners generate marketing images and ad creatives for their products. Simply upload a product image, describe your desired marketing scenes, and let the AI attempt to generate banners, human model images, and ad creatives for various platforms.

## Features

- **Product Banner Generation:**
  - Upload a product image and describe a custom background scene.
  - The AI attempts to generate a photorealistic marketing banner with your product integrated into the scene.

- **Human Model with Product:**
  - Specify age, gender, and ethnicity for a human model.
  - Describe two different poses/interactions with the product.
  - The AI attempts to generate images of the model interacting with your product.

- **Ad Creatives for Multiple Platforms:**
  - Generate ad creatives for Instagram, Facebook, and Website banners.
  - Customize ad text: header, tagline, call-to-action, and tone.

- **Downloadable Results:**
  - Download generated images directly from the app.

## How It Works

1. **Enter your Gemini API Key** in the sidebar (required for AI image generation).
2. **Upload a product image** (PNG or JPG).
3. **Describe the desired marketing scenes** and ad text in the sidebar.
4. **Click "Generate Marketing Content"** to let the AI attempt to create:
    - A product banner with a custom background
    - Two human model images with your product
    - Ad creatives for Instagram, Facebook, and Website
5. **View and download** the generated images and creatives.

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK (`google-generativeai`)
- Pillow

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone this repository**
2. **Install dependencies** (see above)
3. **Get a Gemini API Key** from Google AI Studio
4. **Run the app:**

```bash
streamlit run MarketMintAI/app.py
```

5. **Enter your API key** in the sidebar when prompted.

## Notes & Limitations

- This is a proof-of-concept. The Gemini model is primarily designed for text and multimodal understanding, not for advanced image generation or editing. Results may not always meet expectations for photorealism or text rendering in images.
- For production use, consider integrating with dedicated image editing/generation tools.

## License

Copyright © 2024 Om Kumar. All rights reserved.

This project is for demonstration purposes only.