                                                                                             # ===================================================== 
# 💳 CREDIT CARD FRAUD DETECTION SYSTEM (with Voice + Timeline Graph)
# ---------------------------------------------------------------

# ---------------------- IMPORTS ----------------------
import numpy as np   # Imports NumPy for numerical operations (arrays, math)
import pandas as pd  # Imports Pandas for handling datasets (CSV, tables)
from sklearn.model_selection import train_test_split # train_test_split: Splits your dataset into training and testing sets
from sklearn.linear_model import LogisticRegression # LogisticRegression: Machine learning model used for fraud classification
from sklearn.preprocessing import StandardScaler # StandardScaler: Normalizes/standardizes numeric features for better model accuracy
from sklearn.metrics import accuracy_score # accuracy_score: Calculates how accurate your model predictions are
import streamlit as st # Streamlit: Used to create a web app interface for your ML project
import pyttsx3  # pyttsx3: Used for converting text to speech (voice output in your project)
import matplotlib.pyplot as plt # matplotlib.pyplot: Used to draw graphs (static graph of timeline)
import datetime # datetime: Used to store and display the exact time of each transaction checked

# ---------------------- VOICE FUNCTION ----------------------
def speak(text):
    """Convert prediction result into speech."""
    # This is a docstring — it explains what the function does.
    engine = pyttsx3.init() # Initializes the pyttsx3 text-to-speech engine.
    # Without this, the program cannot generate voice output.
    voices = engine.getProperty('voices') # Fetches the list of all available system voices (male/female).
    engine.setProperty('voice', voices[1].id)  # Selects the second voice from the list → usually the female voice.
    # voices[0] = male, voices[1] = female (most systems)
    engine.setProperty('rate', 175) # Sets the speaking speed of the voice.
    # Lower = slower, higher = faster. 175 is a natural speed.
    engine.say(text) # Adds the text you want to speak to the engine's queue.
    engine.runAndWait()  # Actually speaks the text out loud.
    # Without this line, no sound will come.

# ---------------------- LOAD DATASET ----------------------
# Reading Kaggle creditcard.csv
data = pd.read_csv("creditcard.csv")

# Separate legitimate and fraudulent data
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersampling to balance dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)  # The dataset is highly imbalanced (≈ 492 fraud vs 2.84 lakh legit).
# Undersampling reduces the legit dataset to match the fraud dataset size.
# 'len(fraud)' ensures both fraud and non-fraud have the SAME count.
# This prevents the model from becoming biased towards predicting "non-fraud".
data = pd.concat([legit_sample, fraud], axis=0) # Combines the undersampled legit data + fraud data into ONE balanced dataset.
# axis=0 stacks the rows on top of each other. # Final balanced dataset → ready for model training.

# Features (X) and Label (y)
X = data.drop(columns="Class", axis=1) # Removes the "Class" column from the dataset.
# The remaining columns (V1, V2, ..., V28, Amount, Time) become FEATURES.
# These are the input values the model uses for prediction.
y = data["Class"]  # Extracts the "Class" column as the LABEL.
# y = 0 → Legitimate transaction
# y = 1 → Fraud transaction
# This is the target the model learns to predict.

# ---------------------- FEATURE SCALING ----------------------
scaler = StandardScaler()         #Create the scaler tool.
X_scaled = scaler.fit_transform(X)      #Scale all feature columns to the same range.

# ---------------------- TRAIN / TEST SPLIT ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=2
)  # Split the scaled dataset into training and testing parts. # X_scaled → input features (scaled)
# y → labels (0 = legit, 1 = fraud) # test_size=0.2 → 20% data for testing, 80% for training
# stratify=y → keeps fraud/legit ratio same in both train & test # random_state=2 → ensures same split every time (reproducible)

# ---------------------- MODEL TRAINING ----------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced") # Create Logistic Regression model
# max_iter=1000 → gives model enough time to learn properly # class_weight="balanced" → increases weight of fraud class (because fraud is rare)
model.fit(X_train, y_train)  # Train the model using training data

# ---------------------- ACCURACY ----------------------
train_acc = accuracy_score(model.predict(X_train), y_train)  # Calculate training accuracy → how well model learned training data
test_acc = accuracy_score(model.predict(X_test), y_test)  # Calculate testing accuracy → how well model predicts unseen data

# ---------------------- STREAMLIT PAGE CONFIG ----------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")  # Sets the Streamlit app settings
# page_title → name shown on browser tab # page_icon → tab icon (emoji) # layout="centered" → centers the app layout

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []     # Check if a "history" list exists in session storage # If not, create an empty list
# This history stores all checked transactions

# ---------------------- CUSTOM CSS STYLING ----------------------

# Apply custom HTML/CSS styling to the Streamlit app
# st.markdown() → allows us to inject HTML + CSS into the app
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #e0f7fa, #f1f8ff, #ffffff);
    color: #00334d;
}
[data-testid="stSidebar"] {
    background: #eaf3f9;
    color: #001f33;
}
div.stButton > button:first-child {
    background: linear-gradient(45deg, #4facfe, #00f2fe);
    color: white;
    border-radius: 12px;
    font-weight: bold;
    padding: 0.6rem 1.2rem;
    border: none;
}
h1, h2, h3 {
    color: #003366;
    text-shadow: 1px 1px 2px #b3e5fc;
}
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True → allows HTML/CSS; otherwise Streamlit blocks it for security.

# ---------------------- WELCOME VOICE ----------------------
# This block ensures the welcome voice plays only once when the app starts.
if "voice_played" not in st.session_state:  # Check if "voice_played" does NOT exist in Streamlit's session memory
    speak("Welcome to the Credit Card Fraud Detection System.") # If true, call the speak() function to play welcome message
    st.session_state.voice_played = True  # Store a flag inside session_state so voice does NOT repeat again

# ---------------------- TITLE ----------------------
st.title("💳 Credit Card Fraud Detection System")  # Display main heading/title at the top of the app UI
st.caption("🔐 Detect fraudulent transactions using Logistic Regression (scaled features).") # Caption under the title explaining the purpose (smaller text)

# ---------------------- SIDEBAR MENU ----------------------
# These are the navigation options shown on the left sidebar
menu = ["🏠 Home", "🔎 Predict Transaction", "📊 Graph", "📜 History", "ℹ️ About"]
choice = st.sidebar.radio("Navigate", menu)  # Create a radio button menu in the sidebar # "choice" stores which page the user selected

# ---------------------- HOME PAGE ----------------------
if choice == "🏠 Home":  # This block runs ONLY if the user selects the "Home" page
    st.subheader("📌 Project Overview")  # Display a sub-heading
    st.info("Detect fraudulent credit card transactions using Machine Learning.")  # Blue information box describing the project briefly
    st.write(f"✅ Training Accuracy: **{train_acc:.2f}**")  # Show model accuracy on training data
    st.write(f"✅ Testing Accuracy: **{test_acc:.2f}**")  # Show model accuracy on test data
    st.markdown("💡 *Features are normalized using StandardScaler for improved accuracy.*")  # Add a horizontal separator line

# ---------------------- PREDICTION PAGE ----------------------
# This block runs ONLY when the user selects "Predict Transaction" from the sidebar
elif choice == "🔎 Predict Transaction":
    st.subheader("🔎 Enter Transaction Details")  # Show a sub-heading on the screen

    # User inputs
    card_number = st.text_input("💳 Enter Credit Card Number")  # Text box → user enters the credit card number
    bank_name = st.selectbox("🏦 Select Bank", ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "PNB"])  # Dropdown box → user selects the Bank Name
    validity_year = st.number_input("📅 Card Validity (Year)", min_value=2024, max_value=2035)  # Number input → user selects validity year of the card
    card_limit = st.slider("💰 Credit Limit (₹)", 10000, 500000, step=5000)  # Slider → user selects the credit limit of the card
    transaction_amount = st.slider("💵 Transaction Amount (₹)", 1, int(card_limit), step=100)   # Slider → user selects the transaction amount # max = credit limit
    input_features = st.text_area("Paste 30 features (comma-separated) without Class column")  # Text area → user can paste all 30 ML features

    # Predict Button
    if st.button("🚀 Check Transaction"):  # Button → When clicked, prediction will be made

        if card_number.strip() == "":  # Case: User enters nothing in Card Number
            st.warning("⚠️ Please enter a Credit Card Number")
        else:
            # Case 1: User provided raw 30 features
            if input_features.strip() != "":   # Convert comma-separated values into a list of floats
                arr = [float(x.strip()) for x in input_features.split(",") if x.strip() != ""]
                # Remove Class if included
                if len(arr) == X.shape[1] + 1: # If user mistakenly includes Class column (total 31 values)
                    arr = arr[:-1] # remove last value
                if len(arr) != X.shape[1]:  # If wrong number of features provided
                    st.error(f"❌ Expected {X.shape[1]} features, got {len(arr)}")
                    prediction = [0]  # default = Legitimate
                else:
                    arr_scaled = scaler.transform(np.array(arr).reshape(1, -1)) # Scale features (must match training scale)
                    prediction = model.predict(arr_scaled)  # Predict using trained Logistic Regression model

            # Case 2: 30 features missing → use Amount only for demo
            else:
                features = np.zeros(X.shape[1])  # Create a dummy feature array of zeros
                features[0] = transaction_amount  # Put transaction amount in the first column
                features_scaled = scaler.transform(features.reshape(1, -1))  # Scale dummy features
                prediction = model.predict(features_scaled)  # Predict using only amount (demo prediction)

            # Output result
            result = "✅ Legitimate Transaction" if prediction[0] == 0 else "⚠️ Fraudulent Transaction"  # Decide the text based on prediction output (0 = Safe, 1 = Fraud)

            # Show green success box for safe transaction
            if prediction[0] == 0:
                st.success(result)
                speak("Transaction is safe and legitimate.") # Voice output
            else:  # Show red alert box for fraud transaction
                st.error(result)
                speak("Warning! Fraudulent transaction detected.") # Voice output

            # Save history (with timestamp)
            st.session_state.history.append({
                "Bank": bank_name,
                "Card Number": card_number,
                "Validity": validity_year,
                "Limit": card_limit,
                "Amount": transaction_amount,
                "Result": result,
                "Time": datetime.datetime.now()  # store exact time of prediction
            })

            # Expandable box that shows extra info
            with st.expander("📋 Transaction Details"):  
                # Show all transaction inputs again inside expandable box
                st.write(f"**Bank:** {bank_name}")
                st.write(f"**Card Number:** {card_number}")
                st.write(f"**Validity:** {validity_year}")
                st.write(f"**Limit:** {card_limit}")
                st.write(f"**Amount:** {transaction_amount}")

# ---------------------- TIMELINE GRAPH PAGE ----------------------
# This part runs ONLY when the user selects "📊 Graph" from the sidebar menu
elif choice == "📊 Graph":
    st.subheader("📈 Transaction Check Timeline")  # Display a heading on the screen

    if len(st.session_state.history) == 0:   # If no history exists → user hasn't checked any transactions yet
        st.info("No transactions checked yet.")  # Show information message
    else:  # If at least one transaction exists → draw the graph
        df = pd.DataFrame(st.session_state.history)  # Convert history list into a DataFrame for easy manipulation

        df["Time"] = pd.to_datetime(df["Time"])  # Convert Time column into proper datetime format (important for grouping)
        df["Count"] = 1  # Create a new column "Count" which is always 1 (each row = 1 check)

        # Group by time in 1-minute intervals
        # This tells how many transactions were checked every minute
        timeline = df.groupby(pd.Grouper(key="Time", freq="1min")).sum()  # group per 1 min  # sum the "Count" values

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 4))  # Create a blank graph figure with size 10x4
         # Plot the timeline (x = Time, y = Number of checks)
        ax.plot(
            timeline.index,  # X-axis (time)
            timeline["Count"],  # Y-axis (number of checks)
            linewidth=3,  # thickness of line
            marker="o",   # circle marker at each point
            color="#1f77b4"   # blue color line
        )

        ax.set_title("Transaction Checking Timeline", fontsize=14)  # Graph title
        ax.set_xlabel("Time")   # X-axis label
        ax.set_ylabel("Number of Checks")  # Y-axis label
        ax.grid(True)  # Show grid for better visibility

        st.pyplot(fig)  # Display the graph in Streamlit
        st.success("⬆️ Graph shows WHEN and HOW MANY times you checked transactions.")  # Show success message under the graph

# ---------------------- HISTORY PAGE ----------------------
elif choice == "📜 History":       # If user selects the "History" option in sidebar
    st.subheader("📜 Transaction History")    # Display a heading/title on the page
    if len(st.session_state.history) == 0:     # Check if history list is empty (means no transactions checked yet)
        st.info("No transaction history available.")   # Show an info message if history is empty
    else:
        st.dataframe(pd.DataFrame(st.session_state.history))   # If history is not empty → Convert history list into a DataFrame and display it

# ---------------------- ABOUT PAGE ----------------------
elif choice == "ℹ️ About":       # If user selects the "About" option
    st.subheader("ℹ️ About this Project")   # Show a heading on the page
    # Show a multiline description of your project using Markdown text
    st.write("""
    - Developed as a **B.Tech CSE Machine Learning Project**       
    - Model: **Logistic Regression (Balanced)**                
    - Features scaled with **StandardScaler**              
    - Interactive UI created using **Streamlit**         
    - Includes **Voice Alerts + Timeline Graph**         
    - Dataset: Kaggle (284,807 transactions)           
    """)  
    st.markdown("---")           # Horizontal divider line
    st.markdown("👩‍💻 *Developed by pawan & Payal Baisla*")      # Developer name displayed at bottom

