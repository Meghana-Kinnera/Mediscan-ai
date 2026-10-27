import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info or {}
        self.prompt_template = self.create_prompt_template()

        api_key = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.model = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.0
        )
    
    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            template = """Act like a multidisciplinary team of healthcare professionals.
You will receive a medical report of a patient visited by a Cardiologist, Psychologist, and Pulmonologist.
Task: Review the patient's medical report from the Cardiologist, Psychologist, and Pulmonologist, analyze them and come up with a list of 3 possible health issues of the patient.
Just return a list of bullet points of 3 possible health issues of the patient and for each issue provide the reason.

Cardiologist Report: {cardiologist_report}
Psychologist Report: {psychologist_report}
Pulmonologist Report: {pulmonologist_report}"""
            return PromptTemplate.from_template(template)

        else:
            templates = {
                "Cardiologist": """Act like a cardiologist. You will receive a medical report of a patient.
Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.
Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient's symptoms. Rule out any underlying heart conditions, such as arrhythmias or structural abnormalities, that might be missed on routine testing.
Recommendation: Provide guidance on any further cardiac testing or monitoring needed to ensure there are no hidden heart-related concerns. Suggest potential management strategies if a cardiac issue is identified.
Please only return the possible causes of the patient's symptoms and the recommended next steps.
Medical Report: {medical_report}""",

                "Psychologist": """Act like a psychologist. You will receive a patient's report.
Task: Review the patient's report and provide a psychological assessment.
Focus: Identify any potential mental health issues, such as anxiety, depression, or trauma, that may be affecting the patient's well-being.
Recommendation: Offer guidance on how to address these mental health concerns, including therapy, counseling, or other interventions.
Please only return the possible mental health issues and the recommended next steps.
Patient's Report: {medical_report}""",

                "Pulmonologist": """Act like a pulmonologist. You will receive a patient's report.
Task: Review the patient's report and provide a pulmonary assessment.
Focus: Identify any potential respiratory issues, such as asthma, COPD, or lung infections, that may be affecting the patient's breathing.
Recommendation: Offer guidance on how to address these respiratory concerns, including pulmonary function tests, imaging studies, or other interventions.
Please only return the possible respiratory issues and the recommended next steps.
Patient's Report: {medical_report}"""
            }

            selected_template = templates[self.role]
            return PromptTemplate.from_template(selected_template)
        
    def run(self):
        print(f"{self.role} is running.....")
        if self.role == "MultidisciplinaryTeam":
            cardio_rep = self.extra_info.get('cardiologist_report', '')
            psych_rep = self.extra_info.get('psychologist_report', '')
            pulmo_rep = self.extra_info.get('pulmonologist_report', '')
            prompt = self.prompt_template.format(
                cardiologist_report=cardio_rep,
                psychologist_report=psych_rep,
                pulmonologist_report=pulmo_rep
            )
        else:
            prompt = self.prompt_template.format(medical_report=self.medical_report or '')

        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            err_str = str(e)
            print(f"Error Occurred in {self.role}: {err_str}")
            if "invalid_api_key" in err_str or "401" in err_str:
                return "[Error]: Invalid or missing Groq API Key. Please set GROQ_API_KEY in your environment or .env file."
            return f"[Error in {self.role}]: {err_str}"
    

class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

    
class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

        
class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

    
class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report, 
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)