# -*- coding: utf-8 -*-


!pip install huggingface_hub
!pip install git+https://github.com/huggingface/transformers
!pip install accelerate
!pip install -i https://pypi.org/simple/ bitsandbytes

!pip install -U langchain-huggingface
!pip install langchain_community
!pip install langchain
!pip install sentence-transformers
!pip install pypdf
!pip install ragas
!pip install unstructured[pdf]
!pip install openai
!pip install html2text
!pip install faiss-gpu
!pip install numba
!pip install langchainhub

!pip install owlready2

"""All the necessary imports"""

from google.colab import userdata
userdata.get('HF_TOKEN')

from google.colab import drive
drive.mount('/content/drive')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from tokenize import String

from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from datasets import Dataset, DatasetDict

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_entity_recall,
    answer_similarity,
    summarization_score
)

import pandas as pd
from ragas import evaluate

from owlready2 import *

import gc
import torch

"""SET num VARIABLE BETWEEN 0 AND 13 TO DECIDE WHAT QUERY EXECUTE"""

num = 0

"""SET THE RIGHT PATH TO GET ALL THE NEEDED DATA"""

PATH = "/content/drive/My Drive/Innovazione Digitale/"

"""# MALOnt.owl SECTION"""

MALONT_FILE = PATH + "Cybersecurity domain/MALOnt.owl"
mal_onto = get_ontology(MALONT_FILE).load()

"""TO GET THE INFORMATION FROM ALL HIERICALLY SUPERCLASSES"""

def malSuperclassContext(c):
  context = ""
  if(str(c.is_a[0]) == "owl.Thing"):
    return context
  c = c.is_a[0]
  if len(c.name) > 0 and not c.name in mal_entities:
    context += malGetClassPropertiesContext(c)

  context += malSuperclassContext(c)

  return context

"""TO GET THE INFORMATION FROM ALL HIERICALLY SUBCLASSES"""

def malSubclassContext(c):

  context = malGetClassPropertiesContext(c)

  if(hasattr(c, "subclasses")):
    subclasses = list(c.subclasses())
    for s in subclasses:
      if len(s.name) > 0 and not s.name in mal_entities:
        context += malSubclassContext(s)

  return context

"""TO GET THE INFORMATION FROM THE INDIVIDUALS PROPERTIES AND ITS CLASSES VALUES"""

def malGetClassPropertiesContext(c):
  context = ""
  mal_entities.append(c.name)
  if(len(c.comment) != 0):
    context += c.comment[0] + "\n"

  if(len(c.label) != 0):
    if(hasattr(c, "instances")):
      for i in c.instances():
        for p in i.get_properties():
          v = getattr(i, p.name)
          if(len(v) > 0 and not isinstance(v[0], str)):
            if(v[0].is_a[0] != c and v[0].is_a[0] != owl.Thing):
              if not v[0].is_a[0].name in mal_entities:
                context += malGetClassPropertiesContext(v[0].is_a[0])

  return context

"""THE FUNCTION TO CALL TO OBTAIN ALL THE CONTEXT FROM THE ONTOLOGY"""

def malGetOntologyContext(queryKeywords):
  context = ""
  for keyword in queryKeywords:
    if(mal_onto.search(iri = "http://idea.rpi.edu/malont#*" + keyword + "*") != None):
      classes = mal_onto.search(iri = "http://idea.rpi.edu/malont#*" + keyword + "*")
      for c in classes:
        if(isinstance(c, ThingClass)):
          context += malSubclassContext(c)
          context += malSuperclassContext(c)

  return context

"""# uco_1_5.owl SECTION"""

UCO_FILE = PATH + "Cybersecurity domain/uco_1_5.owl"
uco_onto = get_ontology(UCO_FILE).load()

"""TO GET THE INFORMATION FROM ALL HIERICALLY SUPERCLASSES"""

def ucoSuperclassContext(c):
  context = ""
  if(c.is_a[0].name == "UCOThing"):
    return context
  c = c.is_a[0]
  if len(c.name) > 0 and not c.name in uco_entities:
    context += ucoGetClassPropertiesContext(c)

  context += ucoSuperclassContext(c)

  return context

"""TO GET THE INFORMATION FROM ALL HIERICALLY SUBCLASSES"""

def ucoSubclassContext(c):

  context = ucoGetClassPropertiesContext(c)

  if(hasattr(c, "subclasses")):
    subclasses = list(c.subclasses())
    for s in subclasses:
      if len(s.name) > 0 and not s.name in uco_entities:
        context += ucoSubclassContext(s)

  return context

"""TO GET THE INFORMATION FROM THE PROPERTIES AND ITS CLASSES VALUES"""

def ucoGetClassPropertiesContext(c):
  context = ""
  uco_entities.append(c.name)
  if(len(c.comment) != 0):
    context += c.comment[0] + "\n"

  for p in c.get_class_properties():
    if(p.name != "comment" and p.name != "seeAlso" and p.name != "label"):
      v = getattr(c, p.name)
      if isinstance(v[0], Or):
        mv = str(v[0]).split("|")
        for sv in mv:
          sv = sv.replace(" ", "")
          sv = sv.replace("owl./", "")
          linkedClass = uco_onto.search_one(iri = "http://ffrdc.ebiquity.umbc.edu/ns/ontology/#" + sv)
          if(linkedClass != None and not linkedClass.name in uco_entities):
            context += ucoGetClassPropertiesContext(linkedClass)
      else:
        for value in v:
          if(not value.name in uco_entities):
            context += ucoGetClassPropertiesContext(value)

  return context

"""THE FUNCTION TO CALL TO OBTAIN ALL THE CONTEXT FROM THE ONTOLOGY"""

def ucoGetOntologyContext(queryKeywords):
  context = ""
  for keyword in queryKeywords:
    if(uco_onto.search(iri = "http://ffrdc.ebiquity.umbc.edu/ns/ontology/#*" + keyword + "*") != None):
      classes = uco_onto.search(iri = "http://ffrdc.ebiquity.umbc.edu/ns/ontology/#*" + keyword + "*")
      for c in classes:
        if(isinstance(c, ThingClass)):
          context += ucoSubclassContext(c)
          context += ucoSuperclassContext(c)

  return context

"""# DOCUMENTS AND QUERIES

Loading all documents and preparing them to the RAG system
"""

loader1 = PyPDFLoader(PATH + "Cybersecurity domain/acr2018final.pdf")
pdf1 = loader1.load()
pdf1 = pdf1[5:59]

loader2 = PyPDFLoader(PATH + "Cisco_2017_Midyear_Cybersecurity_Report.pdf")
pdf2 = loader2.load()
pdf2 = pdf2[7:85]

loader3 = PyPDFLoader(PATH + "2024-unit42-incident-response-report.pdf")
pdf3 = loader3.load()
pdf3 = pdf3[6:66]


loader4 = PyPDFLoader(PATH + "Russian_Cyber_Attack_Campaigns.pdf")
pdf4 = loader4.load()


loader5 = PyPDFLoader(PATH + "KL_report_syrian_malware.pdf")
pdf5 = loader5.load()


loader6 = PyPDFLoader(PATH + "cta-cn-2024-0624.pdf")
pdf6 = loader6.load()

urls = ["https://www.lookout.com/threat-intelligence/article/nation-state-mobile-malware-targets-syrians-with-covid-19-lures",
        "https://www.lookout.com/threat-intelligence/article/lookout-discovers-novel-confucius-apt-android-spyware-linked-to-india-pakistan-conflict",
]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

documents = docs_transformed + pdf1 + pdf2 + pdf3 + pdf4 + pdf5 + pdf6
documents

KEYWORDS = []
KEYWORDS = ["" for i in range(14)]

QUERIES = []
QUERIES = ["" for i in range(14)]

GROUND_TRUTHS = []
GROUND_TRUTHS = ["" for i in range(14)]

CHUNK_SIZES = []
CHUNK_SIZES = ["" for i in range(14)]

K = []
K = ["" for i in range(14)]

#queries on MALOnto

KEYWORDS[0] = ["Software", "Virus"]
QUERIES[0] = "What is signature detection used for in antivirus software?"
GROUND_TRUTHS[0] = ("Signature detection is used in antivirus software to search for a unique sequence of bytes that is specific to a piece of malicious code.")
CHUNK_SIZES[0] = 300
K[0] = 1

KEYWORDS[1] = ["Organization", "Country"]
QUERIES[1] = "Which sectors and organizations did RedJuliett target in their cyber attacks in and outside Taiwan?"
GROUND_TRUTHS[1] = ("RedJuliett targeted the technology industry, including organizations in critical technology fields, semiconductor companies, Taiwanese aerospace companies, "
"electronics manufacturers, universities focused on technology, an industrial embedded systems company, a technology-focused research and development institute, and "
"computing industry associations. They also targeted organizations involved in Taiwan's economic and trade policy, including de facto embassies, government departments, "
"think tanks, and a trade promotion organization. Additionally, RedJuliett targeted civil society, including media organizations, a charity, and an NGO focused on human rights. "
"Outside of Taiwan, they targeted government organizations in the Philippines, a government department in Djibouti, and a Malaysian airline.")
CHUNK_SIZES[1] = 2000
K[1] = 1

KEYWORDS[2] = ["Trojan", "Application", "RemoteAccessTrojan-RAT", "Phishing"]
QUERIES[2] = "How do pro-government groups use Trojanized apps on Facebook to infect users, and what techniques do they use for delivery?"
GROUND_TRUTHS[2] = ("Pro-government groups use Trojanized apps on Facebook to infect users by posting links to these apps. They use social engineering techniques to entice users "
"to download and run these apps, such as offering programs for encryption, antivirus protection, and firewall blocking. They also send messages via Skype and share posts on "
"Facebook with links to download these apps. These apps contain Remote Administration Tool (RAT) Trojans, which allow the attackers to control the infected system remotely. "
"The attackers use various RAT variants, including ShadowTech RAT, Xtreme RAT, NjRAT, Bitcomet RAT, Dark Comet RAT, and BlackShades RAT. The attackers continuously evolve their "
"techniques and use different delivery options, such as hiding malicious files in \".scr\" containers to avoid detection by security solutions.")
CHUNK_SIZES[2] = 5000
K[2] = 3

KEYWORDS[3] = ["RemoteAccessTrojan-RAT"]
QUERIES[3] = "What are RemoteAccessTrojan-RAT? What types of RATs did Kaspersky Lab detect in MENA in 2013-2014?"
GROUND_TRUTHS[3] = ("Remote Access Trojans, are a type of malware that give attackers unauthorized control over a victim's computer. The RAT often hides its files deep within the "
"system's directory structure using deceptive filepath names to avoid detection and maintain persistence on the infected system. Kaspersky Lab detected the following types of "
"RATs in MENA in 2013-2014: Zapchast, Bitfrose, Fynloski, and XtremeRAT.")
CHUNK_SIZES[3] = 4000
K[3] = 3

KEYWORDS[4] = ["ThreatActor", "File", "Victim"]
QUERIES[4] = "What are threat actors? How do they exploit Skype messages?"
GROUND_TRUTHS[4] = ("Threat Actors are actual individuals, groups, or organizations believed to be operating with malicious intent. They use social engineering skills, such as "
"the exploitation of Skype messages by sending malicious links or attachments that appear legitimate but they are usually sent from fake or compromised accounts. These links "
"to download programs, like 'SSH VPN' or 'Ammazon Internet Security', can lead to the installation of malware on the victim's device, which can then be used to steal sensitive "
"information or take control of the device.")
CHUNK_SIZES[4] = 2
K[4] = 900

KEYWORDS[5] = ["RemoteAccessTrojan-RAT"]
QUERIES[5] = "What types of RATs did Kaspersky Lab detect in MENA in 2013-2014?"
GROUND_TRUTHS[5] = ("Kaspersky Lab detected the following types of RATs in MENA in 2013-2014: Zapchast, Bitfrose, Fynloski, and XtremeRAT.")
CHUNK_SIZES[5] = 3000
K[5] = 1

KEYWORDS[6] = ["Campaign", "Country", "Attack", "Actor"]
QUERIES[6] = "What is a caimpaign? Which is the country with some of the most notorious actors in cyber attack campaigns? Why?"
GROUND_TRUTHS[6] = ("A campaign is a grouping of adversarial behaviors that describes a set of malicious activities or attacks that occur over a period of time against a specific "
"set of targets. The country with some of the most notorious actors in cyber attack campaigns is Russia. Strategic Russian interests "
"are guided by the desires for Russia to be recognized as a great power, to protect the Russian identity, and to limit global United States power. These themes are evident "
"in components commonly associated with Russian-backed cyber threat campaigns: the weaponization of information through disinformation campaigns and propaganda; attempted "
"interference in democratic processes; strategic positioning within critical infrastructure, perhaps as preparation for potential escalation of hostilities with rival nations.")
CHUNK_SIZES[6] = 1500
K[6] = 1

KEYWORDS[7] = ["Malware", "SunBird"]
QUERIES[7] = "What are SunBird-specific functionality?"
GROUND_TRUTHS[7] = ("SunBird attempts to upload all data it has access to at regular intervals to its command and control (C2) servers. Locally on the infected device, the data "
"is collected in SQLite databases which are then compressed into ZIP files as they are uploaded to C2 infrastructure. SunBird can exfiltrate the following list of data: \n"
"-List of installed applications \n"
"-Browser history \n"
"-Calendar information \n"
"-BlackBerry Messenger (BBM) audio files, documents and images \n"
"-WhatsApp Audio files, documents, databases, voice notes and images \n"
"-Content sent and received via IMO instant messaging application \n"
"SunBird can also perform the following actions: \n"
"-Download attacker specified content from FTP shares \n"
"-Run arbitrary commands as root, if possible \n"
"-Scrape BBM messages and contacts via accessibility services \n"
"-Scrape BBM notifications via accessibility services")
CHUNK_SIZES[7] = 1000
K[7] = 1


#queries on uco ontology

KEYWORDS[8] = ["Defender", "Vendor"]
QUERIES[8] = "What are the challenges that a defender deal with due to the complexity of vendors landscape?"
GROUND_TRUTHS[8] = ("Defenders face several challenges due to the complexity of the vendor landscape: \n"
"- Managing multiple vendors and products adds significant complexity.\n"
"- The difficulty of integrating these products into a cohesive security strategy: they don’t provide an actionable view of the threats.\n"
"- Vulnerabilities due to the lack of coordination between different security tools.\n"
"- It is difficult for defenders to stay up-to-date on the latest security trends and best practices, leading to a potential loss of efficiency and effectiveness.")
CHUNK_SIZES[8] = 2000
K[8] = 3

KEYWORDS[9] = ["DenialOfService"]
QUERIES[9] = "Describe the Permanent Denial of Service attack (PDoS)."
GROUND_TRUTHS[9] = ("Permanent denial of service (PDoS) attacks are rapid bot attacks that render device hardware non-functional, often requiring hardware reinstallation or "
"replacement. These attacks, also known as \"phlashing,\" exploit security flaws to destroy firmware and system functions. The BrickerBot malware uses Telnet brute force to "
"breach devices, corrupts storage with Linux commands, disrupts internet connectivity, and wipes all files on the device.")
CHUNK_SIZES[9] = 1600
K[9] = 1

KEYWORDS[10] = ["Vulnerability", "BufferOverFlow"]
QUERIES[10]= "According to CWE what are the leading errors?"
GROUND_TRUTHS[10] = ("Bufferoverflow is identified by Common Weakness Enumeration (CWE) as the most common type of coding error exploited by criminals highlighting the need for "
" developers to restrict buffers to prevent exploitation.")
CHUNK_SIZES[10] = 400
K[10] = 2

KEYWORDS[11] = ["Mistake", "Defender"]
QUERIES[11] = "What are the systemic issues or mistakes made by defenders that contributed to attackers’ success?"
GROUND_TRUTHS[11] = ("Systemic issues or mistakes by defenders that contribute to attackers' success include: \n"
"- Lack of Preparation: Failing to proactively prepare for compromises results in delayed responses. \n"
"- Insufficient Automation: Over-reliance on manual processes can lead to missed critical alerts and slower response times. \n"
"- Weak Zero Trust Implementation: Not restricting attackers' movement after initial access allows them to cause more damage. \n"
"- Inadequate Defense in Depth: Single-layer defenses without overlapping controls make it easier for attackers to navigate and compromise systems.")
CHUNK_SIZES[11] = 5000
K[11] = 2

KEYWORDS[12] = ["TTP", "ExploitTarget"]
QUERIES[12] = "What are Tactics, Techniques and Procedures (TTPs)? How they are linked with the concept of 'exploit target'?"
GROUND_TRUTHS[12] = ("Tactics, Techniques and Procedures (TTP) are representations of the behavior or modus operandi of cyber adversaries. They can include malware, social "
"engineering techniques, and other means of gaining access to systems or data. Exploit targets are vulnerabilities or weaknesses in software, systems, networks or "
"configurations that are targeted for exploitation by the TTP.")
CHUNK_SIZES[12] = 600
K[12] = 1

KEYWORDS[13] = ["Attack", "Supply chain"]
QUERIES[13]= "Provide a definition of an 'attack'. What type of supply chain attack occurred in September 2017?"
GROUND_TRUTHS[9] = ("An attack is the use of an exploit(s) by an adversary to take advantage of a weakness(s) with the intent of achieving a negative technical impact(s). "
"In September 2017, a supply chain attack occurred in which a software vendor distributed a legitimate software package known as CCleaner, but the binaries contained a Trojan "
"backdoor that was signed with a valid certificate, giving users false confidence that the software they were using was secure. The actors behind this campaign were targeting "
"major technology companies where the software was in use, either legitimately or as part of shadow IT.")
CHUNK_SIZES[13] = 1000
K[13] = 3

num = 7
print(KEYWORDS[num])
print()
print(QUERIES[num])
print()
print(GROUND_TRUTHS[num])
print()
print(CHUNK_SIZES[num])
print()
print(K[num])

"""# Use KEYWORDS to retrieve the ontology entities"""

mal_entities = list()
uco_entities = list()

if(num <= 7):
  ontologyContext = malGetOntologyContext(KEYWORDS[num])
  print()
  print(mal_entities)
else:
  ontologyContext = ucoGetOntologyContext(KEYWORDS[num])
  print()
  print(uco_entities)

ontology_doc = Document(page_content=ontologyContext, metadata={"source": "cybersecurity ontology"})
documents.append(ontology_doc)

"""Changing the query with the ontology entities"""

ONTO_QUERY = QUERIES[num] + "\nThese concepts might be useful to answer the question: \n"
if(num <= 7):
  entities = mal_entities
else:
  entities = uco_entities
for entity in entities:
  ONTO_QUERY += "- " + entity + "\n"
print(ONTO_QUERY)

"""# RAG

Loading the Mistral 7b model: low_cpu_mem_usage=True essential in order to exploit the GPU
"""

#All parameters are meaningful
my_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
my_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True), # For 8 bit quantization
    low_cpu_mem_usage=True,
    max_memory={0:"15GB"}
)

text_generation_pipeline = pipeline(
    model = my_model,
    tokenizer = my_tokenizer,
    task = "text-generation",
    repetition_penalty = 1.1,
    max_new_tokens = 600,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

rag_prompt = hub.pull("rlm/rag-prompt")

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=rag_prompt)

torch.cuda.empty_cache()
gc.collect()

"""Split documents in chunk of strings to improve the RAG context precision (i.e. sentences or pages)"""

text_splitter = CharacterTextSplitter(separator=".", chunk_size=CHUNK_SIZE[num])
chunked_documents = text_splitter.split_documents(documents)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))


# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': K[num]}
)

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

# Run
result = chain.invoke(ONTO_QUERY)

print("CONTEXT:\n")
result['context']

print(result["text"])

"""# RAGAS METRICS EVALUATION

Format the answer
"""

if("```" in result["text"]):
  answer = result["text"].split("```")[1]
else:
  answer = result["text"].split("Answer:")[1]

if("##" in answer):
  answer = result["text"].split("Answer:")[-1]
  if("##" in answer):
    answer = answer.split("##")[0]

answer = answer.replace("\n", " ")

if(not answer.endswith(".")):
  sentences = answer.split(".")
  sentences.remove(sentences[len(sentences)-1])
  answer = ""
  for sentence in sentences:
    answer += sentence + "."
answer=answer.split("Context")[0]
answer=answer.split("Human")[0]
print(answer)

#To keep only the page_content as context

context = ""
for doc in result['context']:
  context += doc.page_content

context

data = {'question': [QUERIES[num]],
        'ground_truth': [GROUND_TRUTHS[num]],
        'answer': [answer],
        'contexts': [[context]],
        'summary': [answer]}

eval_dataset = Dataset.from_dict(data)

dataset_dict = DatasetDict({
    'eval': eval_dataset
})

dataset_dict

import os

os.environ["OPENAI_API_KEY"] =""

import nest_asyncio

nest_asyncio.apply()

results = evaluate(
    dataset_dict["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_entity_recall,
        answer_similarity,
        summarization_score]
)

results

df_metrics = results.to_pandas()
df_metrics.head()

"""# RESULTS ANALYSIS"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

onto_answer_evaluations = pd.read_csv(PATH +"Results/cyber_onto_answer_evaluations.csv", skiprows=2)
onto_answer_evaluations.drop(columns=['Unnamed: 0'], inplace=True)
onto_answer_evaluations

numeric_columns = onto_answer_evaluations.select_dtypes(include=['number'])
column_means = numeric_columns.mean()

print(column_means)

column_std = numeric_columns.std()

print(column_std)

pd.set_option('display.max_colwidth', None)

onto_answer_evaluations.question

onto_answer_evaluations.contexts

onto_answer_evaluations.ground_truth

onto_answer_evaluations.answer