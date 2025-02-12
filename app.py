from flask import Flask, request, jsonify, session
from flask_cors import CORS
import openai
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from flask_session import Session
from io import StringIO
from prophet import Prophet
import traceback


app = Flask(__name__)
CORS(app,supports_credentials=True)  # Enable CORS for frontend access
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
    try:
        selected_workbook_id = session.get('selected_workbook_id')

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

        # ✅ Debugging Output
        print(f"📊 Retrieved Workbook Data: {workbook_data}")

        # OpenAI API Call (Updated for OpenAI v1.0)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "user", "content": query}
            ],
            max_tokens=1500,
            temperature=0
        )

        bot_response = response.choices[0].message.content.strip()

        print(f"🟢 GPT Response: {bot_response}")

        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"🚨 Chatbot Route Error: {str(e)}")
        print(traceback.format_exc())  # Print full error traceback for debugging
        return jsonify({"error": f"500 Internal Server Error: {str(e)}"}), 500





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
