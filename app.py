# app.py
import streamlit as st
import torch
from model import load_model
from tokenizer import load_tokenizer_and_vocab, tokenize_function, char_tokenizer
from utils import clean_text, extract_urls, check_url

def main():
    st.markdown(
        """
        <style>
        .main-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
        }
        .header-title {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .description {
            font-size: 18px;
            margin-bottom: 20px;
            text-align: center;
            color: #555;
        }
        .input-container {
            margin-bottom: 20px;
        }
        .text-area {
            height: 150px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title and Description
    st.markdown('<div class="header-title">Smishing Detection App</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Enter a Bangla SMS text to classify. This application will detect potential smishing messages and analyze any URLs for security threats.</div>', unsafe_allow_html=True)
    
    # Load model and tokenizer
    model = load_model()
    tokenizer, label_encoder, char_vocab = load_tokenizer_and_vocab()

    # Input Text Area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.text_area("Text:", "মাশরাফির সাথে ফ্রি ডিনার করার সুযোগ পেতে ক্যাসিনোতে বাজি ধরুন। শুরু করুন: promusic.co/components/interbank.com/", height=150)
    st.markdown('</div>', unsafe_allow_html=True)


    if st.button('Classify'):
        cleaned_text = clean_text(user_input)
        
        # Create two columns
        col1, col2 = st.columns(2)

        # Column 1: Tokenize and classify
        with col1:
            st.header("Classification Result")

            input_ids = torch.tensor([tokenize_function(cleaned_text, tokenizer)['input_ids']])
            attention_mask = torch.tensor([tokenize_function(cleaned_text, tokenizer)['attention_mask']])
            char_input = torch.tensor([char_tokenizer(cleaned_text, char_vocab)])

            with torch.no_grad():
                logits = model(input_ids, attention_mask, char_input)
                predicted_label = torch.argmax(logits, dim=1)
                predicted_class = label_encoder.inverse_transform(predicted_label.cpu().numpy())[0]
            
            st.success(f'Predicted Class: {predicted_class}')

        # Column 2: Extract and check URLs
        with col2:
            st.header("URL Analysis")

            urls = extract_urls(user_input)
            #st.write(f'Extracted URLs: {urls}')
            
            # Check URLs with VirusTotal
            if urls:
                results = {}
                for url in urls:
                    result = check_url(url)
                    if 'error' in result:
                        results[url] = f"Error: {result['error']}"
                        st.warning(f"**{url}**: {results[url]}")
                    else:
                        results[url] = f"This URL has been checked by {result['total']} security vendors, and {result['positives']} of them flagged it as malicious."
                        if result['is_malicious']:
                            results[url] += " **Warning**: This URL might be a phishing link!"
                            st.warning(f"**{url}**: {results[url]}")
                        else:
                            results[url] += " This URL is likely safe."
                            st.success(f"**{url}**: {results[url]}")


        

if __name__ == "__main__":
    main()