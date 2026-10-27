from flask import Flask, request, render_template, redirect, url_for
from concurrent.futures import ThreadPoolExecutor, as_completed
from Agent import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
import os
import docx

app = Flask(__name__, template_folder='.')
UPLOAD_FOLDER = 'uploads'
RESULT_PATH = 'results/final_diagnosis.txt'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)


def extract_report_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.docx', '.doc']:
        doc = docx.Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return '\n'.join(paragraphs)
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'report' not in request.files or not request.files['report'].filename:
            return render_template("index.html", error="Please choose a file to upload.")

        file = request.files['report']
        filename = file.filename
        allowed_extensions = ('.txt', '.docx', '.doc')

        if filename.lower().endswith(allowed_extensions):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                medical_report = extract_report_text(filepath)
            except Exception as e:
                return render_template("index.html", error=f"Could not read uploaded file: {str(e)}")

            if not medical_report.strip():
                return render_template("index.html", error="The uploaded report file is empty.")

            # Run individual specialists
            agents = {
                "Cardiologist": Cardiologist(medical_report),
                "Psychologist": Psychologist(medical_report),
                "Pulmonologist": Pulmonologist(medical_report)
            }

            responses = {}
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(agent.run): name for name, agent in agents.items()}
                for future in as_completed(futures):
                    agent_name = futures[future]
                    responses[agent_name] = future.result() or "No response returned."

            # Run multidisciplinary agent
            team_agent = MultidisciplinaryTeam(
                cardiologist_report=responses.get("Cardiologist", ""),
                psychologist_report=responses.get("Psychologist", ""),
                pulmonologist_report=responses.get("Pulmonologist", "")
            )
            final_diagnosis = team_agent.run()

            if not final_diagnosis:
                final_diagnosis = "Diagnosis could not be generated."

            with open(RESULT_PATH, 'w', encoding='utf-8') as result_file:
                result_file.write(final_diagnosis)

            return render_template("index.html", diagnosis=final_diagnosis)

        return render_template("index.html", error="Please upload a valid .txt or .docx medical report file.")

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True, port=5000)


#python app.py  http://127.0.0.1:5000