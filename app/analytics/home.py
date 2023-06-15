import streamlit as st
from PIL import Image
from app.analytics import config, cifar_cnn, mri_cnn, cifar_cnn_greenplum
from streamlit_autorefresh import st_autorefresh
import logging

# Initializations
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nanum Gothic');
html, body, [class*="css"]{
   font-family: 'Nanum Gothic';
}
#tanzu-realtime-anomaly-detection-demo{
   color: #6a6161;
}
.blinking {
  animation: blinker 1s linear infinite;
  background: url('https://github.com/agapebondservant/tanzu-realtime-anomaly-detetction/blob/main/app/assets/clock.png?raw=true') no-repeat right;
}

span.predictedlabel{
    font-size: 1.6em;
    color: green;
}

span.metriclabel{
    font-size: 1em;
    color: wheat;
}

@keyframes blinker {
  50% {
    opacity: 0;
  }
}
</style>
""", unsafe_allow_html=True)

st.header('Tanzu/Vmware Imaging Analytics Demo')

st.text('Demonstration of image pattern detection using neutral networks and Vmware Tanzu')

tab1, tab2 = st.tabs(["CIFAR-10", "MRI"])

# CIFAR-10
with tab1:
    uploaded_file = st.file_uploader("Choose an image", key="upl_cifar")
    if uploaded_file is not None:
        cifar_img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(cifar_img, width=200)
        with col2:
            placeholder = st.empty()
            placeholder.header("Loading model...")
            with placeholder.container():
                prediction = cifar_cnn_greenplum.predict(cifar_img, config.model_name, config.model_stage)
                if prediction:
                    st.markdown(f"Predicted Label:<br/> <span class='predictedlabel'>{prediction}</span>",
                                unsafe_allow_html=True)
                    metrics = cifar_cnn.get_metrics()
                    logging.info(f"Metrics = {metrics}")
                    st.markdown(f"<br/>F-1 metric:<br/> <span class='metriclabel'>{metrics.get('f1_score') or 'None available'}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"<br/>Accuracy metric:<br/> <span class='metriclabel'>{metrics.get('accuracy_score') or 'None available'}</span>",
                                unsafe_allow_html=True)
                else:
                    st.header('Please wait...')
                    st.text('(Training is in progress)')

# MRI
with tab2:
    uploaded_file = st.file_uploader("Choose an image", key="upl_mri")
    if uploaded_file is not None:
        mri_img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(mri_img, width=200)
        with col2:
            prediction = mri_cnn.predict(mri_img, config.model_name, config.model_stage)  # TODO: Use In-DB MRI Model
            if prediction:
                st.markdown(f"Predicted Label:<br/> <span class='predictedlabel'>{prediction}</span>",
                            unsafe_allow_html=True)
                metrics = cifar_cnn.get_metrics()
                st.markdown(f"<br/>F-1 metric:<br/> <span class='metriclabel'>{metrics.get('f1_score') or 'None available'}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<br/>Accuracy metric:<br/> <span class='metriclabel'>{metrics.get('accuracy_score') or 'None available'}</span>",
                            unsafe_allow_html=True)
            else:
                st.header('Please wait...')
                st.text('(Training is in progress)')


# Refresh the screen at a configured interval
st_autorefresh(interval=config.refresh_interval * 1000, key="anomalyrefresher")
