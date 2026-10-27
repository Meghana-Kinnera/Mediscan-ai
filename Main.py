import os
import sys
from dotenv import load_dotenv
load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from concurrent.futures import ThreadPoolExecutor, as_completed
from Agent import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
import docx


def extract_report_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.docx', '.doc']:
        doc = docx.Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return '\n'.join(paragraphs)
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()


report_file = "Medical_Report_1.txt"
if len(sys.argv) > 1:
    report_file = sys.argv[1]

if not os.path.exists(report_file):
    print(f"Error: File '{report_file}' not found.")
    sys.exit(1)

medical_report = extract_report_text(report_file)

agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}


def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response


responses = {}

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}

    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response or ""

team_agents = MultidisciplinaryTeam(
    cardiologist_report=responses.get("Cardiologist", ""),
    psychologist_report=responses.get("Psychologist", ""),
    pulmonologist_report=responses.get("Pulmonologist", "")
)

final_diagnosis = team_agents.run() or "No diagnosis generated."
txt_output_path = "results/final_diagnosis.txt"

os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

with open(txt_output_path, "w", encoding="utf-8") as txt_file:
    txt_file.write(final_diagnosis)

print(f"Final Diagnosis has been saved to {txt_output_path}")
print("\n--- FINAL DIAGNOSIS RESULT ---")
print(final_diagnosis)
