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

"""# DOCUMENTS AND QUERIES

Loading all documents and preparing them to the RAG system
"""

loader1 = PyPDFLoader(PATH + "Salmon domain/Fish_length.pdf")
pdf1 = loader1.load()
pdf1 = pdf1[1:]

for page in pdf1:
  page.page_content = page.page_content.replace("Joint Conference on Green Engineering Technology & Applied Computing 2019\nIOP Conf. Series: Materials Science and Engineering 551" +
                                                  " (2019) 012076IOP Publishing\ndoi:10.1088/1757-899X/551/1/012076", "")

loader2 = PyPDFLoader(PATH + "Salmon domain/Salmon.pdf")
pdf2 = loader2.load()
pdf2 = pdf2[:9]

loader3 = PyPDFLoader(PATH + "Salmon domain/Salmon-Industry-Handbook.pdf")
pdf3 = loader3.load()
pdf3 = pdf3[1:124]

urls = ["https://www.marine.ie/site-area/areas-activity/fisheries-ecosystems/salmon-life-cycle",
        "https://www.oceansatlas.org/subtopic/en/c/1303/",
        "https://www.ontario.ca/page/fish-tags#:~:text=A%20fish%20tag%20is%20a,fish%20(known%20as%20internal%20tags)",
        "https://www.oritag.org.za/GettingStarted",
        "https://www.pac.dfo-mpo.gc.ca/fm-gp/salmon-saumon/comm-gear-engin-eng.html",
        "https://fish-commercial-names.ec.europa.eu/fish-names/fishing-gears_en"
        ]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

docs_transformed[0].page_content = docs_transformed[0].page_content[2770:].split("* Contact Us")[0]
docs_transformed[1].page_content = docs_transformed[1].page_content[100:].split("### Related Resources")[0]
docs_transformed[2].page_content = docs_transformed[2].page_content[826:4484]
docs_transformed[3].page_content = docs_transformed[3].page_content[423:].split("This website is managed by")[0]
docs_transformed[4].page_content = docs_transformed[4].page_content[550:].split("Date modified:")[0]
docs_transformed[5].page_content = docs_transformed[5].page_content[830:].split("## Data sources")[0]

docs_5_splits = docs_transformed[5].page_content.split("### Related links\n\n  ")
docs_transformed[5].page_content = ""
for split in docs_5_splits:
  subsplits = split.split("Department.\n\n")
  for sub in subsplits:
    if not sub.startswith("* ©FAO."):
      docs_transformed[5].page_content += sub

documents = docs_transformed + pdf1 + pdf2 + pdf3
documents

"""Define the queries and the relative ground truths"""

QUERIES = []
QUERIES = ["" for i in range(14)]

GROUND_TRUTHS = []
GROUND_TRUTHS = ["" for i in range(14)]

CHUNK_SIZES = []
CHUNK_SIZES = ["" for i in range(14)]

K = []
K = ["" for i in range(14)]

QUERIES[0] = "Give a definition of seine net. What is the purpose of pair seines in fishing?"
GROUND_TRUTHS[0] = ("A seine net is a type of fishing net that hangs vertically in the water with its bottom edge held down by weights and its top edge buoyed by floats. "
"It is a very long net, sometimes featuring a bag in the center, which is deployed either from the shore or from a boat to surround a designated area. The net is operated "
"using two ropes attached to its ends. Pair seines are designed for pair fishing, where two boats operate the net in pair.")
CHUNK_SIZES[0] = 200
K[0] = 2

QUERIES[1] = "What fishing net prevents fish from escaping using netting panels?"
GROUND_TRUTHS[1] = "Surrounding nets and lift nets prevent fish from escaping using netting panels."
CHUNK_SIZES[1] = 200
K[1] = 1

QUERIES[2] = " What are trollers in commercial salmon fisheries?"
GROUND_TRUTHS[2] = "Trollers in commercial salmon fisheries use hooks and lines suspended from large poles extending from the fishing vessel to catch pelagic fishes."
CHUNK_SIZES[2] = 500
K[2] = 3

QUERIES[3] = "What is subsistence fishery? Is there any prohibition in Alaska?"
GROUND_TRUTHS[3] = ("In subistence fishery fishes or other seafood are harvested for noncommercial, customary and traditional uses. These uses include direct personal or "
"family consumption as food, shelter, fuel, clothing, tools, or transportation, for the making and selling of handicraft articles out of nonedible by-products of fish and "
"wildlife resources taken for personal or family consumption, and for the customary trade, barter, or sharing for personal or family consumption. In Alaska, subsistence "
"fisheries may not operate in 'nonsubsistence areas' as designated by the state (AS 16.05.258(c)).")
CHUNK_SIZES[3] = 800
K[3] = 1

QUERIES[4] = "What are the tags implanted in the fish’s abdominal cavity?"
GROUND_TRUTHS[4] = ("The tags implanted in the fish’s abdominal cavity are:\n"
"- PIT tags: Passive Integrated Transponder tags are small, passive radio transponder tag which, when in range, are activated by a signal emitted from a tag reader. "
"The tag then emits a unique identification code back to the reader.\n"
"- Acoustic tags: small, electronic, sound-emitting devices which collects information on fish's physiology and/or movement patterns. Data are transmitted wirelessly, "
"usually through the use of radio waves, acoustic signals or via satellite communication. Reading is done using the tag’s associated equipment, often through a computer connection.\n"
"- Radio tags: emit a radio signal that can be detected by a receiver. Like acoustic tags, radio tags allow researchers to track the movements of tagged fish.\n"
"The tags are inserted into the fish’s abdominal cavity through a needle or surgical wound, which heals and leaves little to no scarring.")
CHUNK_SIZES[4] = 500
K[4] = 1

QUERIES[5] = "Identify and describe the length measurement type used for tuna. What are the automatic methods used for tuna length measurement?"
GROUND_TRUTHS[5] = ("The length measurement type used for tuna is fork length. Fork length is the distance from the tip of the snout to the fork of the tail. It is the "
"most commonly used length measurement type for tuna. Automatic methods used for tuna length measurement are Hough transform, image thinning, best fitting rectangle, "
"Hsiu method and grade-3 polynomial regression. Each method has its own strengths and weaknesses. For example, the Hough transform has a low error rate of less than 5%, "
"while the Hsiu method has a high accuracy rate of measuring smaller fish lengths. However, these methods have not been widely used for tuna length measurement due to "
"their limitations and challenges.")
CHUNK_SIZES[5] = 3000
K[5] = 1

QUERIES[6] = "Summarize the salmon life-cycle's stages."
GROUND_TRUTHS[6] = ("The life stages of salmon are summarized as follows:\n\n"
"Egg (Ova): the salmon begins life as a pea-sized egg, hidden under loose gravel in cool, clean rivers. Eggs have a high mortality rate, with only a "
"small percentage surviving to hatch.\n\n"
"Alevin: upon hatching in spring, the fish are called alevins and have a yolk sac attached, providing nourishment. Once the yolk sac is absorbed, alevins "
"become active, move through the gravel, and must gulp air to fill their swim bladders for neutral buoyancy.\n\n"
"Fry: they have eight fins and feed on microscopic invertebrates during summer; they are typically found in shallow waters near the shoreline.\n\n"
"Parr: in autumn, fry develop into parr; they have vertical stripes and spots for camouflage and they feed on aquatic insects, growing for 1-3 years while "
"establishing territories in the stream.\n\n"
"Juvenile: young fish, mostly similar in form to adult but not yet sexually mature.\n\n"
"Smolt: once they reach 10-25 cm, parr undergo smolting; they become silvery and start swimming with the current, preparing for ocean migration.\n\n"
"Jack: precocial male salmon that have spent one winter less in the ocean than the youngest females of a given species.\n\n"
"Grilse (Adult Salmon): smolts migrated to the ocean that exhibit strong homing instincts to return to their river of origin to spawn.\n\n"
"Kelt: after spawning, salmon are referred to as kelts; weakened from not eating and the energy expended in reproduction, many kelts die.")
CHUNK_SIZES[6] = 7000
K[6] = 1

QUERIES[7] = "List all pacific salmon species. Which is the most widespread species?"
GROUND_TRUTHS[7] = ("The Pacific salmon species are:\n"
"- Chinook salmon (Oncorhynchus tshawytscha): Also known as king salmon or blackmouth salmon, and spring salmon in British Columbia.\n"
"- Chum salmon (Oncorhynchus keta): Known as dog salmon or calico salmon in some parts of the US, and keta in the Russian Far East.\n"
"- Coho salmon (Oncorhynchus kisutch): Also known as silver salmon.\n"
"- Masu salmon (Oncorhynchus masou): Also known as cherry trout in Japan.\n"
"- Pink salmon (Oncorhynchus gorbuscha): Known as humpback salmon or humpies in southeast and southwest Alaska.\n"
"- Sockeye salmon (Oncorhynchus nerka): Also known as red salmon in the US, especially Alaska.\n"
"Chinook salmon is the most widespread species.")
CHUNK_SIZES[7] = 4000
K[7] = 1

QUERIES[8] = "What is aquaculture? How important is it for fish human consumption?"
GROUND_TRUTHS[8] = ("Aquaculture is the culturing of fish, shellfish, aquatic plants, and/or other organisms in captivity or under controlled conditions in the near shore environment. "
"It is an important source of fish for human consumption, providing about half of the world's fish supply. In fact, in 2023, aquaculture accounted for 90 million "
"tonnes (LW) of fish destined for direct human food consumption. Futhermore, aquaculture is a major contributor to the world's economy, with an annual value of around $10 billion.")
CHUNK_SIZES[8] = 400
K[8] = 3

QUERIES[9] = "What are the different regulations for fish farming in Canada based on geographical area?"
GROUND_TRUTHS[9] = ("Fish farming companies in Canada are subject to different regulations depending on the geographical area they operate in. The three primary fish farming areas "
"in Canada are British Columbia, Newfoundland and Labrador, and New Brunswick. In Newfoundland and Labrador and New Brunswick, the Provincial government is the primary regulator "
"and leasing authority. The Province regulates the activity and operations of aquaculture and issues the Aquaculture License, Crown Land lease, and Water Use License where fish "
"farms are located. In Newfoundland and Labrador, the Crown Land Lease for the site is issued for 50 years, the Aquaculture License is issued for 6 years, and the Water Use License "
"is issued for 5 years. In New Brunswick, individual sites are typically granted a lease for 20 years. In British Columbia, Federal and Provincial authorizations are required to "
"operate a marine fish farm site. The Federal Government regulates the activity and operations of aquaculture while the Provincial Government administers the Crown lands where fish "
"farms are located. The Province grants a license to occupy an area of the ocean associated with the individual fish farming site. "
"The tenured encompasses the rearing pens, ancillary infrastructure, and all moorings. Individual site tenures have a specific timeline ranging from five to twenty years.")
CHUNK_SIZES[9] = 3900
K[9] = 1

QUERIES[10] = ("What are the key indicators for projecting future fish harvest volumes? How does smolt release data affect long-term volume estimates? "
"How does seawater temperature affect production cycle length and harvest volumes? How do disease outbreaks impact fish harvest volumes?")
GROUND_TRUTHS[10] = ("The three key indicators for projecting future fish harvest volumes are standing biomass, feed consumption, and smolt release. Standing biomass categorized "
"by size is the best short-term indicator, while standing biomass, feed consumption, and smolt release are good indicators for medium- and long-term harvest. Smolt release data "
"affects long-term volume estimates as it takes up to 2 years from smolt release to harvest. Variation in seawater temperature can impact the length of the production cycle and "
"harvest volumes. A warmer winter can increase harvest volumes for the relevant year, partly at the expense of the subsequent year. Disease outbreaks can impact harvest volume "
"due to mortality and growth slowdown.")
CHUNK_SIZES[10] = 1200
K[10] = 2

QUERIES[11] = "How do husbandry and health practices contribute to maximise salmon survival and fish stock maintenance?"
GROUND_TRUTHS[11] = ("Maximising survival and maintaining healthy fish stocks are primarily achieved through good husbandry and health management practices and policies, which "
"reduce exposure to pathogens and the risk of health challenges. The success of good health management practices has been demonstrated on many occasions and has contributed to "
"an overall improvement in the survival of farmed salmonids. Fish health management plans, veterinary health plans, biosecurity plans, risk mitigation plans, contingency plans, "
"disinfection procedures, surveillance schemes, as well as coordinated and synchronised zone/area management approaches, all support healthy stocks with an emphasis on disease "
"prevention. Prevention of many diseases is achieved through vaccination at an early stage and while the salmon are in freshwater. Vaccines are widely used commercially to reduce "
"the risk of health challenges. With the introduction of vaccines a considerable number of bacterial and viral health issues have been effectively controlled, with the additional "
"benefit that the quantity of licensed medicines prescribed in the industry has been reduced. In some instances medicinal treatment is still required to avoid mortality and for the "
"well-being and welfare of the fish. Even the best managed farms may have to use licensed medicines from time to time, if other measures are not sufficient.")
CHUNK_SIZES[11] = 1400
K[11] = 2

QUERIES[12] = "Why is salmo salar considered a healthy product?"
GROUND_TRUTHS[12] = ("Atlantic salmon is a healthy product because is rich in long-chain omega-3, EPA and DHA, which reduce the risk of cardiovascular disease. Data also "
" indicates that EPA and DHA reduce the risk of a large number of other health issues. Furthermore, it's nutritious, rich in micronutrients, minerals, "
"marine omega-3 fatty acids, high-quality protein and several vitamins.")
CHUNK_SIZES[12] = 1000
K[12] = 1

QUERIES[13] = "What are the main fish stock types?"
GROUND_TRUTHS[13] = ("The main fish stock types are:\n"
"- Acquaculture: the culturing of fish, shellfish, aquatic plants, and/or other organisms in captivity or under controlled conditions in the near shore environment.\n"
"- Hatchery: the artificial breeding, hatching, and rearing through the early life stages of animals -- finfish and shellfish in particular.\n"
"- Wild stock: a stock that is sustained by natural spawning and rearing in the natural habitat, regardless of parentage or origin.")
CHUNK_SIZES[13] = 1600
K[13] = 1

print(QUERIES[num])
print()
print(GROUND_TRUTHS[num])
print()
chunk_size="Setted chunk size: " + str(CHUNK_SIZES[num])
print(chunk_size)
print("Setted k: " + str(K[num]))

"""# RAG

Loading the Mistral 7b model
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

# define the prompt template
rag_prompt = hub.pull("rlm/rag-prompt")

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=rag_prompt)

# Command to partially empty the GPU
torch.cuda.empty_cache()
gc.collect()

"""Split documents in chunk of strings to improve the RAG context precision (i.e. sentences)"""

text_splitter = CharacterTextSplitter(separator=".", chunk_size=CHUNK_SIZES[num])
chunked_documents = text_splitter.split_documents(documents)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': K[num]}
)

"""RETRIEVE THE CONTEXT AND GENERATE THE ANSWER"""

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

# Run
result = chain.invoke(QUERIES[num])

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
  answer_parts = result["text"].split("Answer:")
  if(len(answer_parts) >= 3):
    answer = answer_parts[2]
  else:
    answer = answer.split("## Answer")[1]
  if("##" in answer):
    answer = answer.split("##")[0]

answer = answer.replace("\n", " ")

if(not answer.endswith(".")):
  sentences = answer.split(".")
  sentences.remove(sentences[len(sentences)-1])
  answer = ""
  for sentence in sentences:
    answer += sentence + "."

answer = answer.split("Human")[0]
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

os.environ["OPENAI_API_KEY"] = "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"

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

answer_evaluations = pd.read_csv(PATH + "Results/salmon_answer_evaluations.csv", skiprows=2)
answer_evaluations

numeric_columns = answer_evaluations.select_dtypes(include=['number'])
column_means = numeric_columns.mean()

print(column_means)

column_std = numeric_columns.std()

print(column_std)

answer_evaluations.question

answer_evaluations.contexts

answer_evaluations.ground_truth

answer_evaluations.answer

!pip freeze