from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
import os, dotenv, json, signal, sys, yaml
from openai import AzureOpenAI
import pandas as pd

# Load environment variables
dotenv.load_dotenv()

# Function to handle interruption
def signal_handler(sig, frame):
    print('Interrupted, saving progress...')
    df.to_csv('output_progress.csv', index=False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Azure and OpenAI configurations
openai_gpt4_api_base = os.getenv("AZURE_OPENAI_GPT4_API_BASE")
openai_gpt4_api_key = os.getenv("AZURE_OPENAI_GPT4_API_KEY")
openai_gpt4_deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
openai_api_type = os.getenv("AZURE_OPENAI_API_TYPE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
search_endpoint = os.getenv("AZURE_AI_SEARCH_API_BASE")
search_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
sabrina_knowledge_index = os.getenv("AZURE_AI_SEARCH_INDEX")
sabrina_knowledge_semantic_configuration = os.getenv("AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION")
APIM_SUBSCRIPTION_KEY = os.getenv("APIM_SUBSCRIPTION_KEY")
APIM_ENDPOINT = os.getenv("APIM_API_BASE")
openai_gpt35_deployment_name = os.getenv("OPENAI_GPT35_DEPLOYMENT_NAME")

credential = AzureKeyCredential(search_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=sabrina_knowledge_index, credential=credential)
evaluator_client = AzureOpenAI(
    azure_endpoint=openai_gpt4_api_base,
    api_key=openai_gpt4_api_key,
    api_version=openai_api_version
)

# Asynchronous Azure OpenAI Client Setup with API Management
qna_client = AzureOpenAI(
    default_headers={"Ocp-Apim-Subscription-Key": APIM_SUBSCRIPTION_KEY},
    api_key=APIM_SUBSCRIPTION_KEY,
    azure_endpoint=APIM_ENDPOINT,
    azure_deployment=openai_gpt35_deployment_name + "/extensions",
    api_version=openai_api_version,
)

def search_documents(user_query: str):
    results = search_client.search(search_text=user_query, top=10, query_type=QueryType.SEMANTIC, semantic_configuration_name=sabrina_knowledge_semantic_configuration)
    search_results = {idx + 1: {'content': result['content'].replace('\n', ' '),
                                'metadata_storage_name': result['metadata_storage_name']}
                      for idx, result in enumerate(results)}
    return search_results

# Function to load the YAML file
def load_system_prompt(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['system_prompt']

def generate_qna_response(user_input):
    # Load the system prompt from the YAML file
    system_prompt = load_system_prompt('system_prompt.yaml')
    completion = qna_client.chat.completions.create(
        model=openai_gpt35_deployment_name,
        messages=[{"role": "user", "content": user_input}],
        extra_body = {
            "dataSources": [
                {
                    "type": "AzureCognitiveSearch",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "key": search_key,
                        "indexName": sabrina_knowledge_index,
                        "topNDocuments": 10,
                        "strictness": 4,
                        "fields_mapping": {
                            "content_fields_separator": "\n",
                            "content_fields": [
                                "content",
                            ],
                            "filepath_field": "metadata_storage_name",
                        },
                        "roleInformation": system_prompt,
                        "queryType": "semantic",
                        "semanticConfiguration": sabrina_knowledge_semantic_configuration,
                    }
                }
            ]
        },
        temperature = 0,
        max_tokens = 1000,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0,
        stop = None,
        seed = 42
    )

    qna_response = completion.choices[0].message.content

    # Extract the context (contains citations)
    context = completion.choices[0].message.context['messages'][0]['content']

    print(context)

    context_data = json.loads(context)

    # Extract the `content` and `filepath` (metadata_storage_name) fields from each citation
    context_list = [{"content": entry["content"], "metadata_storage_name": entry["filepath"]}
                    for entry in context_data.get("citations", [])]

    return context_list, qna_response

def evaluate_response(user_query, knowledge_available):
    system_prompt = f"""Task: Evaluate if the provided user query has a corresponding knowledge source in the database that matches exactly or very closely.

User Query: "{user_query}"

List of Available Knowledge Sources:
{knowledge_available}

Instructions: Compare the user query with the list of available knowledge sources. Be mindful that some knowledge sources may appear similar but are distinctly different (e.g., "requirements to open a bank account" vs. "how to open a bank account"). If a knowledge source precisely or very closely matches the user query, return the number of that source from the list. If no single source can fully answer the query, return 'None'.

Output: Return the result in JSON format.

Example:
///
User Query: "What are the requirements to open a bank account?"
List of Available Knowledge Sources:
1. Guide on how to open a bank account
2. Checklist of requirements for opening a bank account
3. Overview of different types of bank accounts

If the knowledge in item 2 can answer the user query, the output should be:
{{"result": "2"}}

If no knowledge source is adequate, the output should be:
{{"result": "None"}}///"""  # Your current system_prompt
    messages = [{"role": "system", "content": system_prompt}]
    completion = evaluator_client.chat.completions.create(
        model=openai_gpt4_deployment_name,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=100,
        seed=42
    )
    # print("===============\n",system_prompt, "===============\n")
    # return json.loads(completion.choices[0].message.content)
    raw_content = completion.choices[0].message.content
    # print("Raw EVALUATOR JSON Content:", raw_content)  # Add this line to debug
    return json.loads(raw_content)

def judge_response(user_query, chosen_knowledge, knowledge_available):
    system_prompt = f"""Task: Evaluate the result from the first agent regarding the relevance and precision of the selected knowledge source to the user query. Also, verify if the decision of 'None' (no relevant knowledge source) by the first agent is accurate, especially considering the subtleties between similar but distinct knowledge sources.

Instructions:
1. If the first agent has chosen a knowledge source, evaluate whether this source can indeed answer the user query with high precision, recognizing subtle differences in content (e.g., "requirements" vs. "methods"). If the chosen knowledge source does not precisely match the user query in terms of addressing the exact question asked, provide a detailed explanation why the chosen knowledge source is not relevant.
2. If multiple knowledge sources are needed to answer the user query fully, acknowledge that this is acceptable.
3. If the first agent's output was 'None', verify by checking all available knowledge sources to determine if there truly is no relevant and precise source for the user query.

Input from Agent 1:
{{
  "user_query": "{user_query}",
  "chosen_knowledge": {chosen_knowledge}
}}

List of Available Knowledge Sources:
{knowledge_available}

Output: Return the evaluation result in JSON format.

Example:
Input from Agent 1:
{{
  "user_query": "What are the requirements to open a bank account?",
  "chosen_knowledge": 2
}}

List of Available Knowledge Sources:
1. Guide on how to open a bank account
2. Checklist of requirements for opening a bank account
3. Overview of different types of bank accounts

If the chosen knowledge (item 2) is correct, the output should be:
{{"evaluation": "Confirmed"}}

If the chosen knowledge is incorrect, the output should be:
{{"evaluation": "Incorrect", "reason": "The chosen source does not address the specific requirements which are distinct from the procedural steps outlined in the query."}}

If no knowledge source is adequate, including a verification of 'None':
{{"evaluation": "None confirmed"}}"""  # Your current system_prompt
    messages = [{"role": "system", "content": system_prompt}]
    completion = evaluator_client.chat.completions.create(
        model=openai_gpt4_deployment_name,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=1000,
        seed=42
    )
    # return json.loads(completion.choices[0].message.content)
    raw_content = completion.choices[0].message.content
    # print("\nRaw JUDGES JSON Content:", raw_content, "\n")  # Add this line to debug
    return json.loads(raw_content)

# Initialize DataFrame with explicit data types for problematic columns
column_defaults = {
    'metadata_storage_name': pd.Series(dtype='object'),
    'ground_truth': pd.Series(dtype='object'),
    'flag': pd.Series(dtype='object')
}
# Load or create DataFrame
try:
    df = pd.read_csv("test.csv", dtype={key: 'object' for key in column_defaults})
except FileNotFoundError:
    df = pd.DataFrame(column_defaults)

def main():
    for df_index, row in df.iterrows():
        try:
            user_query = row['user_query']
            context_list, qna_response = generate_qna_response(user_query)
            
            # Prepare the initial knowledge base from the generated context
            knowledge_available = "\n".join([f"{idx + 1}. {item['content']}" for idx, item in enumerate(context_list)])
            attempt_count = 0
            max_attempts = 3
            chosen_knowledge_contents = []
            chosen_metadata_names = []

            while attempt_count < max_attempts:
                eval_response = evaluate_response(user_query, knowledge_available)
                chosen_knowledges = eval_response.get("result", "None")
                chosen_knowledge_contents = []
                chosen_metadata_names = []

                print("CHOSEN KNOWLEDGE: ", chosen_knowledges)

                # Handle multiple chosen knowledge sources
                if ',' in chosen_knowledges:
                    for index in chosen_knowledges.split(','):
                        index = index.strip()
                        if index.isdigit() and int(index) <= len(context_list):
                            chosen_knowledge_contents.append(context_list[int(index) - 1]['content'])
                            chosen_metadata_names.append(context_list[int(index) - 1]['metadata_storage_name'])
                elif chosen_knowledges.isdigit() and int(chosen_knowledges) <= len(context_list):
                    chosen_knowledge_contents.append(context_list[int(chosen_knowledges) - 1]['content'])
                    chosen_metadata_names.append(context_list[int(chosen_knowledges) - 1]['metadata_storage_name'])
                else:
                    chosen_knowledge_contents.append("None")

                if not chosen_metadata_names:
                    chosen_metadata_names = None

                chosen_knowledge_content = " | ".join(chosen_knowledge_contents)
                print("CHOSEN KNOWLEDGE CONTENT: ", chosen_knowledge_content)

                judge_eval = judge_response(user_query, chosen_knowledge_content, knowledge_available)

                if judge_eval.get("evaluation") != "Incorrect":
                    break
                else:
                    reason_clean = judge_eval.get("reason", "").replace('\n', '\\n').replace('"', '\\"')
                    new_system_prompt = f"\n\nPlease re-evaluate based on this feedback. Feedback number {attempt_count + 1}:\nPrevious chosen knowledge: {chosen_knowledge_content}\nFeedback: {reason_clean}"
                    knowledge_available += new_system_prompt
                    attempt_count += 1

            if attempt_count == max_attempts and judge_eval.get("evaluation") == "Incorrect":
                judge_eval["evaluation"] = "None confirmed"

            df.at[df_index, 'metadata_storage_name'] = ", ".join(chosen_metadata_names) if chosen_metadata_names else None
            df.at[df_index, 'ground_truth'] = chosen_knowledge_content if chosen_knowledge_contents else ""
            df.at[df_index, 'flag'] = "knowledge available" if judge_eval.get("evaluation") == "Confirmed" else "knowledge not available"
            print(f"Processed row {df_index + 1}: Flag = {df.at[df_index, 'flag']}")

        except Exception as e:
            print(f"Error processing row {df_index + 1}: {e}")
            continue  # Skip to the next row on error

    df.to_csv('test_output.csv', index=False)
    print("Processing completed and saved to output_final.csv.")

if __name__ == "__main__":
    main()

