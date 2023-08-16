import pickle
import streamlit as st
import pandas as pd
from PIL import Image


model_file = 'GradientBoostingClassifier().bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

def main():
    #image = Image.open('images/icone.jpeg')
    image2 = Image.open('images/download.jpeg')
    #st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch")
    )

    st.sidebar.info('Predict Shipwreck Survival')
    st.sidebar.image(image2)
    st.title("Shipwreck Survival Predictor")
    if add_selectbox == 'Online':
        Title = st.selectbox('Please enter your title:', ['Mr', 'Mrs', 'Miss', 'Master', 'Rare', 'Royal'])
        Embarked  = st.selectbox(' Select your destination :', ['S', 'C', 'Q'])
        Sex  = st.selectbox(' Please select your gender:', ['male', 'female'])
        Pclass = st.number_input('Please enter the Seat Class :', min_value=1, max_value=3, value=1)
        Fare = st.number_input('Please enter the Ticket Price :', min_value=0, max_value=5000, value=0)
        SibSp = st.number_input('Number of Siblings/Spouse aboard :', min_value=0, max_value=10, value=0)
        Parch = st.number_input('Number of Parent/Children aboard :', min_value=0,max_value=10, value=0)
        Age = st.number_input('Please enter  your Age :', min_value=0, max_value=100, value=0)
        output = ""
        output_prob = ""
        input_dict = {
            'Sex': Sex,
            'Embarked' :Embarked,
            'Title':Title,
            'Pclass': Pclass,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare' : Fare
        }

    if st.button("Predict"):
        X = dv.transform([input_dict])
        y_pred = model.predict_proba(X)[0, 1]
        Survived = y_pred >= 0.6
        output_prob = float(y_pred)
        output = bool(Survived)
    st.success('Survive: {0}, Survival Probability: {1}'.format(output, output_prob))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = dv.transform([data])
            y_pred = model.predict_proba(X)[0, 1]
            Survived = y_pred >= 0.6
            Survived = bool(Survived)
            st.write(Survived)

if __name__ == '__main__':
    main()


#############        streamlit run main.py       ###################