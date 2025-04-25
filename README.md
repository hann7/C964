# Diabetic Retinopathy Severity Detector

This web application uses a deep learning model to predict the severity of diabetic retinopathy from retinal fundus images.


## User Guide

### 1. Clone or Download the Project
Clone this repository or download the ZIP and extract it.

```bash
git clone https://github.com/your-username/your-repo-name.git
```

---

### 2. Install Python 3.10
Ensure you have **Python 3.10** installed.  
Visit [python.org](https://www.python.org/downloads/release/python-3100/)  
Scroll to the bottom of the page to the **“Files”** section and download the appropriate installer for your operating system and architecture.

---

### 3. Install Dependencies
Navigate to the project root in your terminal and run:

```bash
pip install -r requirements.txt
```

---

### 4. Start the Application
Still in the project root, launch the app by running:

```bash
python -m app.server.app
```

You should see output indicating that the app is running at:  
`http://127.0.0.1:5000/`

---

### 5. Use the Application
1. Open a browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
2. Upload a retinal image using the **file selector**
3. Click **Submit**
4. View the predicted severity in a red-highlighted box
5. Scroll down to explore data visualizations

---


## Dataset Info

The model was trained on a subset of the **Kaggle Diabetic Retinopathy Detection** dataset. Labels represent severity levels:

- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

---


