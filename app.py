from flask import Flask, request, jsonify, session
from flask_cors import CORS
import openai
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from flask_session import Session
from io import StringIO
from prophet import Prophet

from flask import Flask, session
from flask_session import Session
from flask_cors import CORS

app = Flask(__name__)

# ✅ Correctly configure session storage
app.config["SESSION_TYPE"] = "filesystem"  # Store sessions in a persistent filesystem
app.config["SESSION_PERMANENT"] = False  # Ensure session does not expire immediately
app.config["SESSION_USE_SIGNER"] = True  # Sign session cookies for security
app.config["SESSION_COOKIE_SAMESITE"] = "None"  # Allow cross-origin session sharing
app.config["SESSION_COOKIE_SECURE"] = True  # Required for HTTPS
app.config["SECRET_KEY"] = "6f1e22b0a1b34d6fa9b2d5e3f8c48e14bcef3e9246b5d6d2b678901c345af8d6"  # Replace with a strong secret key

Session(app)
CORS(app, supports_credentials=True)

app.secret_key = "d39c892a9ef547d2917a12c3e3e1bd078f7ef3a9ffb2edb6dd65a12cf8f2f61a"

# OpenAI API Key
openai.api_key = "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"  # Replace with your OpenAI API key

# Tableau API Configuration
TABLEAU_SERVER = "https://dub01.online.tableau.com"
SITE_ID = "b14c2718-c023-42c8-8411-39950d2e57d6"
CONTENT_URL = "harsh-f6134b2d0d"
PAT_NAME = "chatbot"
PAT_SECRET = "jO4ctmdARPyykt3GGrd2rQ==:k5q5DwNjZuh51NXC9Rm6XJ7DYgXm2DFV"
API_VERSION = "3.24"


def get_tableau_auth_token():
    url = f"{TABLEAU_SERVER}/api/{API_VERSION}/auth/signin"
    headers = {"Content-Type": "application/json"}
    payload = {
        "credentials": {
            "personalAccessTokenName": PAT_NAME,
            "personalAccessTokenSecret": PAT_SECRET,
            "site": {"contentUrl": "harsh-f6134b2d0d"}  # Ensure correct site content URL
        }
    }

    response = requests.post(url, json=payload, headers=headers)

    # ✅ Print the raw response for debugging
    print(f"\n🔍 Tableau Auth Request Sent to: {url}")
    print(f"🟡 Request Payload: {payload}")
    print(f"🟡 Response Status Code: {response.status_code}")
    print(f"🟡 Raw Response:\n{response.text}\n")

    if response.status_code != 200:
        print(f"⚠️ Failed to get auth token: {response.text}")
        return None

    try:
        xml_response = ET.fromstring(response.text)

        credentials_element = xml_response.find("{http://tableau.com/api}credentials")

        if credentials_element is not None and "token" in credentials_element.attrib:
            auth_token = credentials_element.attrib["token"]
            print(f"✅ Auth Token Retrieved: {auth_token}")
            return auth_token
        else:
            print("⚠️ Token not found in XML response!")
            return None

    except ET.ParseError as e:
        print(f"❌ XML Parsing Error: {e}")
        return None



def fetch_tableau_data(workbook_id):
    """Fetch all data inside the workbook (views, sheets, tables)."""
    token = get_tableau_auth_token()
    if not token:
        return None, "Tableau Authentication Failed"

    url = f"{TABLEAU_SERVER}/api/{API_VERSION}/sites/{SITE_ID}/workbooks/{workbook_id}/views"
    headers = {"X-Tableau-Auth": token, "Accept": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"⚠️ Failed to get views: {response.text}")
        return None, f"Tableau API Error: {response.text}"

    views_data = response.json()
    views = views_data.get("views", {}).get("view", [])

    if not views:
        print("⚠️ No views found in workbook!")
        return None, "No data views found inside the workbook."

    all_data = {}

    for view in views:
        view_id = view["id"]
        view_name = view["name"]
        data_url = f"{TABLEAU_SERVER}/api/{API_VERSION}/sites/{SITE_ID}/views/{view_id}/data"

        response = requests.get(data_url, headers=headers)

        if response.status_code == 200:
            csv_data = response.text.strip()

            if not csv_data:
                print(f"⚠️ No data returned for {view_name}")
                all_data[view_name] = "⚠️ No data available for this view."
                continue  # Skip to the next view

            print(f"\n🟡 Raw CSV Data for {view_name}:\n{csv_data[:500]}")  # Print only first 500 characters

            try:
                df = pd.read_csv(StringIO(csv_data))

                if df.empty:
                    print(f"⚠️ DataFrame is empty for {view_name}")
                    all_data[view_name] = "⚠️ No data available for this view."
                    continue

                print(f"\n✅ DataFrame Extracted for {view_name}:\n{df.head()}")

                all_data[view_name] = df.to_dict(orient="records")  # Convert dataframe to dictionary format

            except pd.errors.EmptyDataError:
                print(f"🚨 ERROR: No columns to parse for {view_name}")
                all_data[view_name] = "⚠️ No data available (Empty response)."

    if not all_data:
        return None, "⚠️ No data available in this workbook."

    return all_data, None


def generate_forecast(data, periods=30):
    """
    Generate a forecast using Prophet.
    
    Parameters:
        data (list of dicts): Data from Tableau, expecting "ds" (date) and "y" (value).
        periods (int): Number of future days to predict.

    Returns:
        dict: Forecasted data with upper/lower confidence intervals.
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Ensure correct column names for Prophet
        df.rename(columns={"date": "ds", "value": "y"}, inplace=True)

        # Initialize Prophet model
        model = Prophet()
        model.fit(df)

        # Create future dataframe for predictions
        future = model.make_future_dataframe(periods=periods)

        # Generate forecast
        forecast = model.predict(future)

        # Select only the relevant columns
        forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

        # Convert to JSON format
        return forecast_result.to_dict(orient="records")

    except Exception as e:
        print(f"🚨 Forecasting Error: {str(e)}")
        return {"error": "Forecasting failed"}
    

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # ✅ Ensure request has JSON body
        data = request.get_json()
        if not data or "workbook_id" not in data:
            return jsonify({"error": "No workbook_id provided"}), 400

        selected_workbook_id = data["workbook_id"]  # Get workbook ID from request
        print(f"🟢 Debug: Received Workbook ID → {selected_workbook_id}")

        workbook_data, error = fetch_tableau_data(selected_workbook_id)

        if error:
            return jsonify({"error": error}), 500

        if not workbook_data:
            return jsonify({"error": "No relevant data found in Tableau."})

        # Extract time-series data (Ensure "date" and "value" exist)
        for view_name, records in workbook_data.items():
            time_series_data = [
                {"ds": record["date"], "y": record["value"]}
                for record in records
                if "date" in record and "value" in record
            ]

            if time_series_data:
                forecast_result = generate_forecast(time_series_data, periods=30)
                return jsonify({"forecast": forecast_result})

        return jsonify({"error": "No valid time-series data found for forecasting."})

    except Exception as e:
        return jsonify({"error": f"500 Internal Server Error: {str(e)}"}), 500


@app.route('/chatbot', methods=['POST'])
def chatbot():
    selected_workbook_id = session.get('selected_workbook_id')
    print(f"⚠️ Debug: Session Data at chatbot request → {session}")
    if not selected_workbook_id:
        return jsonify({"error": "No workbook selected"}), 400

    query = request.json.get("query", "").lower()
    print(f"\n📩 Query: {query}")
    print(f"🔍 Fetching Data for Workbook ID: {selected_workbook_id}")

    workbook_data, error = fetch_tableau_data(selected_workbook_id)

    if error:
        return jsonify({"error": error}), 500

    if not workbook_data:
        return jsonify({"response": "No relevant data found in Tableau."})

    # ✅ Step 1: Check for Time-Series Data (Forecasting)
    for view_name, records in workbook_data.items():
        print(f"📊 Checking {view_name} for forecasting...")

        time_series_data = [
            {"ds": record["date"], "y": record["value"]}
            for record in records
            if "date" in record and "value" in record
        ]
    
        if time_series_data:
            print(f"✅ Time-series data detected in {view_name}: {time_series_data[:5]}")  # Print first 5 records for debugging
            forecast_result = generate_forecast(time_series_data, periods=30)
            print(f"🔮 Forecast generated: {forecast_result[:5]}")  # Print first 5 forecasted points
            return jsonify({"forecast": forecast_result})

    print("⚠️ No time-series data found for forecasting!")

    # ✅ Step 2: Otherwise, Process General Query
    filtered_data = {}

    for view_name, records in workbook_data.items():
        if "name starts with" in query:  
            letter = query.split("starts with")[-1].strip().upper()[0]
            filtered_records = [record for record in records if "User" in record and record["User"].startswith(letter)]
            filtered_data[view_name] = filtered_records
        else:
            filtered_data[view_name] = records  

    MAX_RECORDS = 60  
    data_str = ""

    for view_name, records in filtered_data.items():
        limited_records = records[:MAX_RECORDS]
        data_str += f"\nView: {view_name}\n"
        for record in limited_records:  
            data_str += f"{record}\n"  

    print(f"\n📊 Data Sent to GPT (Limited to {MAX_RECORDS} records per view):\n{data_str}")

    # ✅ Step 3: Construct Messages for OpenAI
    messages = [
        {"role": "system", "content": "You are a data analyst answering queries based on Tableau workbook data."},
        {"role": "user", "content": f"Data:\n{data_str}\n\nQuestion: {query}\nAnswer:"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500,
            temperature=0
        )

        answer = response["choices"][0]["message"]["content"].strip()
        print(f"🟢 GPT Response: {answer}")

        return jsonify({"response": answer})

    except openai.RateLimitError as e:
        print(f"🚨 OpenAI Rate Limit Error: {str(e)}")
        return jsonify({"response": "Sorry, I can't process this request due to token limits. Try a simpler query. 🥰"})




@app.route('/get-workbooks', methods=['GET'])
def get_workbooks():
    token = get_tableau_auth_token()
    if not token:
        return jsonify({"error": "Authentication failed"}), 401

    url = f"{TABLEAU_SERVER}/api/{API_VERSION}/sites/{SITE_ID}/workbooks"
    headers = {"X-Tableau-Auth": token}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch workbooks: {response.text}")
        return jsonify({"error": "Failed to fetch workbooks"}), response.status_code

    try:
        xml_response = ET.fromstring(response.text)
        workbooks = []
        
        # Extract workbook details from XML
        for workbook in xml_response.findall(".//{http://tableau.com/api}workbook"):
            workbook_data = {
                "id": workbook.attrib.get("id"),
                "name": workbook.attrib.get("name"),
            }
            workbooks.append(workbook_data)

        if not workbooks:
            print("⚠️ No workbooks found.")
            return jsonify({"error": "No workbooks found"}), 404

        return jsonify({"workbooks": workbooks})

    except ET.ParseError as e:
        print(f"❌ XML Parsing Error: {e}")
        return jsonify({"error": "Failed to parse Tableau workbook response"}), 500


@app.route('/select-workbook', methods=['POST'])
def select_workbook():
    workbook_id = request.json.get('workbook_id')

    if not workbook_id:
        return jsonify({"error": "No workbook ID provided"}), 400

    session['selected_workbook_id'] = workbook_id  # ✅ Store workbook in session
    print(f"✅ Workbook Stored in Flask Session: {session.get('selected_workbook_id')}")

    return jsonify({"message": "Workbook selected successfully", "workbook_id": workbook_id})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
