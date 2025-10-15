import joblib
import pandas as pd

# ✅ Load model
model = joblib.load("model.pkl")

# ✅ New data to predict
new_data = pd.DataFrame({
    "size": [1200, 2500],
    "bedrooms": [2, 4]
})
preds = model.predict(new_data)

# ✅ Generate HTML
html = "<h2>Predictions</h2><ul>"
for i, p in enumerate(preds):
    html += f"<li>House {i+1}: ₹{int(p):,}</li>"
html += "</ul>"

# ✅ Save to GitHub Pages output folder
with open("output/predictions.html", "w") as f:
    f.write(html)
