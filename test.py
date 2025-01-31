import requests

response = requests.post(
    "https://sos-detection-api.onrender.com/predict",
    json={
        "acceleration": 111,
        "rotation": 10,
        "magnetic_field": 300,
        "light": 73.2
    }
)
print(response.json())