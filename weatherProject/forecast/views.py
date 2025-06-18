from django.shortcuts import render
import requests
import pytz
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta

# views.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
API_KEY = os.getenv('OPENWEATHER_API_KEY')


def get_custom_icon(condition, is_day=True):
    """Map weather conditions to custom day/night icons"""
    condition = condition.lower()
    suffix = "day" if is_day else "night"

    if "clear" in condition:
        return f"clear-sky-{suffix}.png"
    elif "partly cloudy" in condition:
        return f"partly-cloudy-{suffix}.png"
    elif "cloud" in condition:
        return f"cloudy-{suffix}.png"
    elif "rain" in condition or "drizzle" in condition:
        return f"rainy-{suffix}.png"
    elif "thunder" in condition or "storm" in condition:
        return f"thunderstorm-{suffix}.png"
    elif "snow" in condition:
        return f"snowy-{suffix}.png"
    elif "mist" in condition or "fog" in condition or "haze" in condition:
        return f"mist-{suffix}.png"

    # Default icon
    return f"cloudy-{suffix}.png"


# Load ML models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_models")
MODELS = {}

try:
    MODELS["temp"] = joblib.load(os.path.join(MODEL_DIR, "temp_model.pkl"))
    print("Loaded temperature model")
except:
    print("Temperature model not found")

try:
    MODELS["min_temp"] = joblib.load(os.path.join(MODEL_DIR, "min_temp_model.pkl"))
    print("Loaded min temperature model")
except:
    print("Min temperature model not found")

try:
    MODELS["max_temp"] = joblib.load(os.path.join(MODEL_DIR, "max_temp_model.pkl"))
    print("Loaded max temperature model")
except:
    print("Max temperature model not found")

try:
    MODELS["humidity"] = joblib.load(os.path.join(MODEL_DIR, "humidity_model.pkl"))
    print("Loaded humidity model")
except:
    print("Humidity model not found")


def get_current_weather(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return {
            "error": f"City not found: {response.json().get('message', 'Unknown error')}"
        }

    data = response.json()

    # Calculate chance of rain (if available)
    rain_chance = 0
    if "rain" in data and "1h" in data["rain"]:
        rain_volume = data["rain"]["1h"]
        rain_chance = min(100, int(rain_volume * 20))

    description = data.get("weather", [{}])[0].get("description", "")

    # Detect day or night based on icon code
    icon_code = data.get("weather", [{}])[0].get("icon", "01d")
    is_day = icon_code.endswith("d")

    icon = get_custom_icon(description, is_day=is_day)

    return {
        "city": data.get("name"),
        "country": data.get("sys", {}).get("country"),
        "current_temp": round(data["main"].get("temp", 0)),
        "feels_like": round(data["main"].get("feels_like", 0)),
        "description": description,
        "pressure": data["main"].get("pressure"),
        "humidity": data["main"].get("humidity"),
        "wind_speed": data["wind"].get("speed"),
        "clouds": data.get("clouds", {}).get("all", 0),
        "sunrise": data.get("sys", {}).get("sunrise"),
        "sunset": data.get("sys", {}).get("sunset"),
        "icon_code": icon_code,
        "rain_chance": rain_chance,
        "icon": icon,
    }


def predict_weather(current_weather, days_ahead):
    """Predict weather using ML models"""
    predictions = []
    today = datetime.utcnow().date()

    for i in range(1, days_ahead + 1):
        target_date = today + timedelta(days=i)

        # Prepare features for prediction
        features = {
            "year": target_date.year,
            "month": target_date.month,
            "day": target_date.day,
            "day_of_year": target_date.timetuple().tm_yday,
            "MinTemp": current_weather["current_temp"] - 3,  # Estimated min
            "MaxTemp": current_weather["current_temp"] + 3,  # Estimated max
            "Humidity": current_weather["humidity"],
            "Pressure": current_weather["pressure"],
            "WindGustSpeed": current_weather["wind_speed"],
        }

        features_df = pd.DataFrame([features])

        pred = {}
        if "min_temp" in MODELS:
            pred["min_temp"] = MODELS["min_temp"].predict(features_df)[0]
        if "max_temp" in MODELS:
            pred["max_temp"] = MODELS["max_temp"].predict(features_df)[0]
        if "temp" in MODELS:
            pred["temp"] = MODELS["temp"].predict(features_df)[0]
        if "humidity" in MODELS:
            pred["humidity"] = MODELS["humidity"].predict(features_df)[0]

        if i == 1:
            day_name = "Tomorrow"
        else:
            day_name = target_date.strftime("%A")

        avg_temp = (
            pred.get("min_temp", current_weather["current_temp"])
            + pred.get("max_temp", current_weather["current_temp"])
        ) / 2

        temp = pred.get("temp", avg_temp)
        humidity = pred.get("humidity", current_weather["humidity"])

        # Determine icon and description based on conditions
        if humidity > 70 and temp < 10:
            icon_name = get_custom_icon("snow", is_day=True)
            description = "Snow"
        elif humidity > 60 and temp < 20:
            icon_name = get_custom_icon("rain", is_day=True)
            description = "Rain"
        elif humidity > 40 and temp < 25:
            icon_name = get_custom_icon("drizzle", is_day=True)
            description = "Drizzle"
        elif temp > 25:
            icon_name = get_custom_icon("clear", is_day=True)
            description = "Clear Sky"
        elif temp > 15:
            icon_name = get_custom_icon("partly cloudy", is_day=True)
            description = "Partly Cloudy"
        else:
            icon_name = get_custom_icon("cloud", is_day=True)
            description = "Cloudy"

        predictions.append(
            {
                "day_name": day_name,
                "temp_min": round(pred.get("min_temp", avg_temp - 3)),
                "temp_max": round(pred.get("max_temp", avg_temp + 3)),
                "temp": round(temp),
                "humidity": round(humidity),
                "description": description,
                "icon": icon_name,
            }
        )

    return predictions


def get_api_forecast(city, api_key):
    """Get 5-day forecast from API"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    data = response.json()
    forecast_list = []
    today = datetime.utcnow().date()
    processed_dates = set()

    for forecast in data["list"]:
        forecast_time = datetime.strptime(forecast["dt_txt"], "%Y-%m-%d %H:%M:%S")
        forecast_date = forecast_time.date()
        description = forecast["weather"][0]["description"]
        icon_code = forecast["weather"][0]["icon"]
        is_day = icon_code.endswith("d")
        icon = get_custom_icon(description, is_day)

        if forecast_date <= today or forecast_date in processed_dates:
            continue

        if forecast_time.hour >= 12:  # Use midday forecast
            days_diff = (forecast_date - today).days
            day_name = "Tomorrow" if days_diff == 1 else forecast_date.strftime("%A")

            forecast_list.append(
                {
                    "day_name": day_name,
                    "temp_min": round(forecast["main"]["temp_min"]),
                    "temp_max": round(forecast["main"]["temp_max"]),
                    "temp": round(forecast["main"]["temp"]),
                    "description": description,
                    "rain_chance": round(forecast.get("pop", 0) * 100),
                    "icon": icon,
                }
            )
            processed_dates.add(forecast_date)

            if len(forecast_list) >= 5:
                break

    return forecast_list


def get_enhanced_forecast(city, api_key, days=7):
    current_weather = get_current_weather(city, api_key)

    if "error" in current_weather:
        return []

    api_forecast = get_api_forecast(city, api_key)
    ml_forecast = predict_weather(current_weather, days)

    combined_forecast = []

    for i in range(days):
        if i < len(api_forecast):
            # Use API data as is (with actual descriptions and icons)
            combined_forecast.append(api_forecast[i])
        else:
            # Use ML forecast for remaining days
            combined_forecast.append(ml_forecast[i])

    return combined_forecast


def get_hourly_forecast(city, api_key):
    """Get hourly forecast data for the next 48 hours"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    data = response.json()
    hourly_list = []
    now = datetime.utcnow()

    for forecast in data["list"]:
        forecast_time = datetime.strptime(forecast["dt_txt"], "%Y-%m-%d %H:%M:%S")
        description = forecast["weather"][0]["description"]
        icon_code = forecast["weather"][0]["icon"]
        is_day = icon_code.endswith("d")
        icon = get_custom_icon(description, is_day)

        # Include forecasts for the next 48 hours
        if (forecast_time - now).total_seconds() > 0:
            time_str = forecast_time.strftime("%I %p").lstrip("0")

            hourly_list.append(
                {
                    "time": time_str,
                    "temp": round(forecast["main"]["temp"]),
                    "description": description,
                    "rain_chance": round(forecast.get("pop", 0) * 100),
                    "icon": icon,
                }
            )

            # Stop after 16 data points to maintain performance
            if len(hourly_list) >= 16:
                break

    return hourly_list


def get_temperature_graph_data(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    graph_data = {"labels": [], "values": []}

    # Get current time
    now = datetime.utcnow()

    # Get forecasts for the next 7 days
    for forecast in data["list"]:
        # Convert forecast time to datetime
        forecast_time = datetime.strptime(forecast["dt_txt"], "%Y-%m-%d %H:%M:%S")

        # Only include forecasts for the next 7 days
        if (forecast_time - now).total_seconds() > 0:
            # Format time as "Tue 3 PM" or "Wed 9 AM"
            time_str = forecast_time.strftime("%a %I %p").lstrip("0")

            # Add to graph data
            graph_data["labels"].append(time_str)
            graph_data["values"].append(round(forecast["main"]["temp"]))

        # Only take 16 data points to avoid overcrowding
        if len(graph_data["labels"]) >= 16:
            break

    return graph_data


def get_current_weather_by_coords(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Location not found"}
    return response.json()


def home(request):
    context = {}
    if request.method == "POST":
        city = request.POST.get("city")
        if not city:
            context["error"] = "Please enter a city name"
        else:
            weather_data = get_current_weather(city, API_KEY)

            if "error" not in weather_data:
                # Convert timestamps to readable times
                timezone = pytz.timezone("Asia/Kathmandu")

                sunrise_dt = datetime.fromtimestamp(
                    weather_data["sunrise"], tz=timezone
                )
                sunset_dt = datetime.fromtimestamp(weather_data["sunset"], tz=timezone)

                # Get enhanced 7-day forecast
                forecast_data = get_enhanced_forecast(city, API_KEY, days=8)

                # Get hourly forecast data
                hourly_data = get_hourly_forecast(city, API_KEY)

                # Get temperature data for graph
                temperature_data = get_temperature_graph_data(city, API_KEY)

                context = {
                    **weather_data,
                    "current_time": datetime.now(timezone).strftime(
                        "%b %d, %Y %I:%M %p"
                    ),
                    "sunrise_time": sunrise_dt.strftime("%I:%M %p"),
                    "sunset_time": sunset_dt.strftime("%I:%M %p"),
                    "forecasts": forecast_data,
                    "hourly_forecast": hourly_data,
                    "temperature_data": temperature_data,
                }
            else:
                context["error"] = weather_data["error"]

    if request.method == "POST":
        lat = request.POST.get("lat")
        lon = request.POST.get("lon")
        city = request.POST.get("city")

        if lat and lon:
            # User sent coordinates, get weather by lat/lon
            weather_json = get_current_weather_by_coords(lat, lon, API_KEY)

            if "error" not in weather_json:
                # Extract city name from weather_json for further forecasts
                city_name = weather_json.get("name")

                # Use your existing function to parse weather data from JSON response
                weather_data = {
                    "city": weather_json.get("name"),
                    "country": weather_json.get("sys", {}).get("country"),
                    "current_temp": round(weather_json["main"].get("temp", 0)),
                    "feels_like": round(weather_json["main"].get("feels_like", 0)),
                    "description": weather_json.get("weather", [{}])[0].get(
                        "description", ""
                    ),
                    "pressure": weather_json["main"].get("pressure"),
                    "humidity": weather_json["main"].get("humidity"),
                    "wind_speed": weather_json["wind"].get("speed"),
                    "clouds": weather_json.get("clouds", {}).get("all", 0),
                    "sunrise": weather_json.get("sys", {}).get("sunrise"),
                    "sunset": weather_json.get("sys", {}).get("sunset"),
                    "icon_code": weather_json.get("weather", [{}])[0].get(
                        "icon", "01d"
                    ),
                    "rain_chance": int(weather_json.get("rain", {}).get("1h", 0) * 20),
                    "icon": get_custom_icon(
                        weather_json.get("weather", [{}])[0].get("description", ""),
                        is_day=weather_json.get("weather", [{}])[0]
                        .get("icon", "01d")
                        .endswith("d"),
                    ),
                }

                timezone = pytz.timezone("Asia/Kathmandu")
                sunrise_dt = datetime.fromtimestamp(
                    weather_data["sunrise"], tz=timezone
                )
                sunset_dt = datetime.fromtimestamp(weather_data["sunset"], tz=timezone)

                forecast_data = get_enhanced_forecast(city_name, API_KEY, days=8)
                hourly_data = get_hourly_forecast(city_name, API_KEY)
                temperature_data = get_temperature_graph_data(city_name, API_KEY)

                context = {
                    **weather_data,
                    "current_time": datetime.now(timezone).strftime(
                        "%b %d, %Y %I:%M %p"
                    ),
                    "sunrise_time": sunrise_dt.strftime("%I:%M %p"),
                    "sunset_time": sunset_dt.strftime("%I:%M %p"),
                    "forecasts": forecast_data,
                    "hourly_forecast": hourly_data,
                    "temperature_data": temperature_data,
                }

            else:
                context["error"] = weather_json["error"]

        elif city:
            # Your existing city code here
            weather_data = get_current_weather(city, API_KEY)

            if "error" not in weather_data:
                timezone = pytz.timezone("Asia/Kathmandu")
                sunrise_dt = datetime.fromtimestamp(
                    weather_data["sunrise"], tz=timezone
                )
                sunset_dt = datetime.fromtimestamp(weather_data["sunset"], tz=timezone)

                forecast_data = get_enhanced_forecast(city, API_KEY, days=8)
                hourly_data = get_hourly_forecast(city, API_KEY)
                temperature_data = get_temperature_graph_data(city, API_KEY)

                context = {
                    **weather_data,
                    "current_time": datetime.now(timezone).strftime(
                        "%b %d, %Y %I:%M %p"
                    ),
                    "sunrise_time": sunrise_dt.strftime("%I:%M %p"),
                    "sunset_time": sunset_dt.strftime("%I:%M %p"),
                    "forecasts": forecast_data,
                    "hourly_forecast": hourly_data,
                    "temperature_data": temperature_data,
                }
            else:
                context["error"] = weather_data["error"]

        else:
            context["error"] = "Please enter a city name or allow location access."

    return render(request, "home.html", context)