import numpy as np
import pandas as pd
import streamlit as st


def main_streamlit_demos():
    st.header('Streamlit Demos')

    if st.sidebar.checkbox('Table demo'):
        df = pd.DataFrame({
            'first column': [1, 2, 3, 4],
            'second column': [10, 20, 30, 40]
        })
        st.write("Here's our first attempt at using data to create a table:")
        st.write(df)

        option = st.sidebar.selectbox(
            'Which number do you like best?',
            df['first column'])

        st.write('You selected: %r' % option)

    if st.sidebar.checkbox('Chart demo'):
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c'])

        st.line_chart(chart_data)

    if st.sidebar.checkbox('Map demo'):
        map_data = pd.DataFrame(
            np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
            columns=['lat', 'lon'])

        st.map(map_data)

    if st.sidebar.checkbox('Button demo'):
        if st.button('Say hello'):
            st.write('Why hello there')
        else:
            st.write('Goodbye')
