import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.predict import predict_specialty

st.set_page_config(page_title="MediRoute AI", page_icon="🩺", layout="centered")

st.title("🩺 MediRoute AI")
st.markdown("""
Welcome to **MediRoute AI**. Describe how you are feeling naturally in the box below. Our NLP engine will extract your symptoms and our machine learning model will recommend the most appropriate medical specialist.
*Note: This is an AI-powered triage tool and does not replace professional medical advice.*
""")

st.divider()

symptoms_input = st.text_area(
    "How are you feeling today?", 
    placeholder="e.g., I have a pimple on my face and my whole body is itching...",
    height=120
)

consent = st.checkbox("I understand that this tool provides AI-generated suggestions and is not a formal medical diagnosis.")

if st.button("Analyze Symptoms", type="primary", use_container_width=True):
    if not symptoms_input.strip():
        st.warning("Please tell us how you are feeling before analyzing.")
    elif not consent:
        st.error("You must agree to the medical disclaimer before proceeding.")
    else:
        with st.spinner("Extracting symptoms using NLP..."):
            try:
                results = predict_specialty(symptoms_input)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    # Show the patient what the AI actually "heard"
                    st.info(f"🔍 **Symptoms Detected by AI:** {', '.join(results['extracted_symptoms'])}")
                    
                    st.subheader("👨‍⚕️ Primary Recommendation")
                    st.success(f"**{results['predicted_specialty']}**")
                    
                    confidence_pct = results['confidence_score'] * 100
                    st.metric("Model Confidence", f"{confidence_pct:.1f}%")
                    
                    st.divider()
                    
                    st.subheader("📊 Top 3 Specialty Matches")
                    for item in results['top_3_predictions']:
                        sp = item['specialty']
                        prob = item['probability'] * 100
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(float(item['probability']))
                        with col2:
                            st.write(f"**{sp}** ({prob:.1f}%)")
                            
            except FileNotFoundError:
                st.error("Model file not found! Please ensure you trained the model locally before running Docker.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")