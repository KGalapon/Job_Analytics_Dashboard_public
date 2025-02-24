import streamlit as st
import pickle
import re
import pyLDAvis
import pyLDAvis.gensim
from streamlit import components


st.set_page_config(
    page_title="Greetings",
    page_icon="👋",
    layout = 'wide'
)

# Streamlit app
st.title("Job Description Topic Clusters")
st.write('''
         The visualization below shows the major groupings of job descriptions across the database. 
        Nearer blobs means more similar topics. The chart on the right indicates which terms per topic are the most salient/important.
         Clustering was performed using the [Latent Dirichlet Allocation topic model.](%s)
         ''' % 'https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation')

with open("lda_vis.html", "r", encoding="utf-8") as f:
    html_string = f.read()

components.v1.html(html_string,width=1300,height=800, scrolling=False)