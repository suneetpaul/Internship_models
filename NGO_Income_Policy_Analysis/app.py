"""
app.py — Income Bracket Estimator
Renders a custom HTML/CSS form (via components.html) for the inputs, but the
actual prediction is computed by the real trained model.pkl — not duplicated
JS math. The HTML form submits by setting the page's query params, which
Streamlit reads on rerun to run pipeline.predict_proba() and show the result.

Run: streamlit run app.py
Needs model.pkl and model_metadata.pkl in the same folder.
"""

import json

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Income Estimator", page_icon="💰", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

pipeline = keras.models.load_model(MODEL_PATH)
metadata = joblib.load("model_metadata.pkl")

education_options = sorted(metadata["education_map"].items(), key=lambda kv: kv[1])
education_years_by_label = dict(education_options)

st.markdown(
    "<style>#MainMenu, header, footer {visibility:hidden;} "
    ".block-container{padding-top:2rem; max-width:680px;} "
    ".stApp{background:#ffffff;}</style>",
    unsafe_allow_html=True,
)


def options_html(values, selected=None):
    return "".join(
        f'<option value="{v}"{" selected" if v == selected else ""}>{v}</option>'
        for v in values
    )


def education_options_html(selected_label="HS-grad"):
    return "".join(
        f'<option value="{label}"{" selected" if label == selected_label else ""}>{label}</option>'
        for label, _years in education_options
    )


# ---------------------------------------------------------------------------
# The custom HTML/CSS form. On submit, JS builds a query string and sets
# window.parent.location.search so the *outer* Streamlit page reruns with
# the profile in st.query_params.
# ---------------------------------------------------------------------------
form_html = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  :root{{
    --ink:#18181b; --ink-soft:#3f3f46; --ink-faint:#71717a; --line:#e4e4e7;
  }}
  *{{box-sizing:border-box;}}
  body{{margin:0; background:#ffffff; font-family:'Inter',sans-serif; color:var(--ink);}}

  .wrap{{ padding:8px 4px 24px; }}
  h1{{ font-size:1.5rem; font-weight:700; margin:0 0 2px; }}
  .subtitle{{ color:var(--ink-faint); font-size:0.92rem; margin:0 0 1.8rem; }}

  label{{ display:block; font-weight:500; color:var(--ink-soft); font-size:0.83rem; margin-bottom:6px; }}
  .field{{ margin-bottom:16px; }}
  .row{{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}

  select, input[type="number"]{{
    width:100%; font-family:'Inter',sans-serif; font-size:0.9rem; color:var(--ink);
    background:#fff; border:1px solid var(--line); border-radius:8px; padding:9px 10px;
    appearance:none; -webkit-appearance:none;
  }}
  select{{
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6'><path d='M0 0L5 6L10 0Z' fill='%2371717a'/></svg>");
    background-repeat:no-repeat; background-position:right 12px center; padding-right:30px;
  }}
  select:focus, input:focus{{ outline:none; border-color:var(--ink); }}

  .slider-row{{ display:flex; align-items:center; gap:12px; }}
  input[type="range"]{{ flex:1; accent-color:var(--ink); height:4px; }}
  .slider-val{{ font-size:0.86rem; font-weight:600; min-width:28px; text-align:right; color:var(--ink-soft); }}

  hr{{ border:none; border-top:1px solid var(--line); margin:1.5rem 0; }}

  button{{
    width:100%; background:var(--ink); color:#fff; border:none; border-radius:8px;
    padding:12px 0; font-family:'Inter',sans-serif; font-weight:600; font-size:0.93rem;
    cursor:pointer; margin-top:0.3rem;
  }}
  button:hover{{ background:#27272a; }}
</style>

<div class="wrap">
  <h1>Income Bracket Estimator</h1>
  <p class="subtitle">Fill in a profile to estimate the chance of earning above $50,000/year.</p>

  <form id="predict-form">
    <div class="row">
      <div class="field">
        <label for="age">Age</label>
        <div class="slider-row">
          <input type="range" id="age" min="17" max="90" value="38" step="1" />
          <span class="slider-val" id="age-out">38</span>
        </div>
      </div>
      <div class="field">
        <label for="sex">Sex</label>
        <select id="sex">{options_html(metadata['categories']['sex'], 'Male')}</select>
      </div>
    </div>

    <div class="row">
      <div class="field">
        <label for="race">Race</label>
        <select id="race">{options_html(metadata['categories']['race'], 'White')}</select>
      </div>
      <div class="field">
        <label for="marital">Marital status</label>
        <select id="marital">{options_html(metadata['categories']['marital_status'], 'Married')}</select>
      </div>
    </div>

    <div class="field">
      <label for="continent">Native region</label>
      <select id="continent">{options_html(metadata['categories']['native_contienent'], 'north_america')}</select>
    </div>

    <hr/>

    <div class="field">
      <label for="education">Education level</label>
      <select id="education">{education_options_html('HS-grad')}</select>
    </div>

    <div class="row">
      <div class="field">
        <label for="workclass">Work class</label>
        <select id="workclass">{options_html(metadata['categories']['workclass'], 'Private')}</select>
      </div>
      <div class="field">
        <label for="occupation">Occupation</label>
        <select id="occupation">{options_html(metadata['categories']['occupation'], 'Adm-clerical')}</select>
      </div>
    </div>

    <div class="field">
      <label for="hours">Hours worked per week</label>
      <div class="slider-row">
        <input type="range" id="hours" min="1" max="99" value="40" step="1" />
        <span class="slider-val" id="hours-out">40</span>
      </div>
    </div>

    <hr/>

    <div class="row">
      <div class="field">
        <label for="capgain">Capital gain ($)</label>
        <input type="number" id="capgain" min="0" max="99999" value="0" step="100" />
      </div>
      <div class="field">
        <label for="caploss">Capital loss ($)</label>
        <input type="number" id="caploss" min="0" max="4356" value="0" step="50" />
      </div>
    </div>

    <button type="submit">Estimate income bracket</button>
  </form>
</div>

<script>
  const ageSlider = document.getElementById("age");
  const ageOut = document.getElementById("age-out");
  ageSlider.addEventListener("input", () => {{ ageOut.textContent = ageSlider.value; }});

  const hoursSlider = document.getElementById("hours");
  const hoursOut = document.getElementById("hours-out");
  hoursSlider.addEventListener("input", () => {{ hoursOut.textContent = hoursSlider.value; }});

  document.getElementById("predict-form").addEventListener("submit", function(e) {{
    e.preventDefault();
    const params = new URLSearchParams({{
      submitted: "1",
      age: ageSlider.value,
      sex: document.getElementById("sex").value,
      race: document.getElementById("race").value,
      marital: document.getElementById("marital").value,
      continent: document.getElementById("continent").value,
      education: document.getElementById("education").value,
      workclass: document.getElementById("workclass").value,
      occupation: document.getElementById("occupation").value,
      hours: hoursSlider.value,
      capgain: document.getElementById("capgain").value || "0",
      caploss: document.getElementById("caploss").value || "0"
    }});
    // Update the OUTER Streamlit page's URL so Python sees these via st.query_params
    const topUrl = window.parent.location.origin + window.parent.location.pathname + "?" + params.toString();
    window.parent.location.href = topUrl;
  }});
</script>
"""

components.html(form_html, height=900, scrolling=True)


# ---------------------------------------------------------------------------
# Read submitted profile from query params and run the REAL model
# ---------------------------------------------------------------------------
qp = st.query_params

if qp.get("submitted") == "1":
    education_label = qp.get("education", "HS-grad")
    education_years = education_years_by_label.get(education_label, 9)

    input_df = pd.DataFrame([{
        "age": int(qp.get("age", 38)),
        "education_no_of_years": education_years,
        "capital_gain": int(qp.get("capgain", 0)),
        "capital_loss": int(qp.get("caploss", 0)),
        "working_hours_per_week": int(qp.get("hours", 40)),
        "workclass": qp.get("workclass", "Private"),
        "marital_status": qp.get("marital", "Married"),
        "occupation": qp.get("occupation", "Adm-clerical"),
        "race": qp.get("race", "White"),
        "sex": qp.get("sex", "Male"),
        "native_contienent": qp.get("continent", "north_america"),
    }])

    proba = pipeline.predict_proba(input_df)[0][1]
    pct_above = round(proba * 100, 1)
    pct_below = round(100 - pct_above, 1)
    is_above = proba >= 0.5

    if is_above:
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                    padding:1.4rem 1.6rem;margin-top:0.4rem;display:flex;
                    align-items:center;justify-content:space-between;
                    font-family:'Inter',sans-serif;">
            <span style="font-size:0.95rem;color:#3f3f46;font-weight:500;">Likely earns above $50,000</span>
            <span style="font-size:1.8rem;font-weight:700;color:#16a34a;">{pct_above}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#fafaf9;border:1px solid #e7e5e4;border-radius:10px;
                    padding:1.4rem 1.6rem;margin-top:0.4rem;display:flex;
                    align-items:center;justify-content:space-between;
                    font-family:'Inter',sans-serif;">
            <span style="font-size:0.95rem;color:#3f3f46;font-weight:500;">Likely earns $50,000 or below</span>
            <span style="font-size:1.8rem;font-weight:700;color:#57534e;">{pct_below}%</span>
        </div>
        """, unsafe_allow_html=True)