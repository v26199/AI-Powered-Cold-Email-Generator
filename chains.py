import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.1-70b-versatile"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `company name`, `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res

    def write_mail(self, cname, role, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            Dear [Hiring Manager Name],

            My name is Vishal Patel, and I am an experienced Data Scientist with a strong focus on machine learning, NLP, and generative LLM models. 
            I am writing to express my interest in the {Job_title} position at {Company_Name}, which I discovered through your company's career job posting.

            With over 5 years of experience in data science and machine learning, I believe my background makes me an excellent fit for this role.

            My recent project involved developing an advanced LLM-based cold email generator that automated job listing extraction and personalized email generation. 
            This innovation resulted in a 30% increase in response rates and a 25% rise in project engagement, demonstrating my ability to drive significant improvements in outreach and efficiency.

            My expertise aligns perfectly with the skills required for this role. 
            I specialize in Python, machine learning, and NLP, with a proven track record of optimizing AI solutions and deploying them effectively. 
            I am adept at leveraging modern technologies to address complex challenges and enhance business processes.

            For your convenience, I have included the most relevant portfolio examples from the following links to showcase my capabilities:
            * {link_list}

            I would welcome the opportunity to discuss how my skills and experience align with your needs. I look forward to the possibility of contributing to your team.

            Best regards,

            Vishal Patel
            Data Scientist

            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "Job_title": role,
            "Company_Name": cname,
            "link_list": links
        })
        return res.content


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
