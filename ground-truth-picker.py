from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
import os, dotenv, json
from openai import AzureOpenAI

openai_api_type = os.getenv("AZURE_OPENAI_API_TYPE")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Azure OpenAI Setup GPT-4
openai_gpt4_api_base = os.getenv("AZURE_OPENAI_GPT4_API_BASE")
openai_gpt4_api_key = os.getenv("AZURE_OPENAI_GPT4_API_KEY")
openai_gpt4_deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")

# Azure AI Search Semantic Configuration
semantic_configuration_name = os.getenv("AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION")

def get_search_client() -> SearchClient:
    """
    Initializes and returns an instance of the Azure SearchClient.

    :return: An instance of Azure SearchClient configured with environment variables.
    """
    search_service_name = os.getenv("SEARCH_SERVICE_NAME")
    index_name = os.getenv("SABRINA_KNOWLEDGE_INDEX")
    api_key = os.getenv("AI_SEARCH_API_KEY")
    endpoint = f"https://{search_service_name}.search.windows.net/"

    credential = AzureKeyCredential(api_key)
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

def search_documents(current_ground_truth: str):
    # Get the Azure SearchClient instance
    search_client = get_search_client()
    
    # Perform the search, specifying the top parameter to limit results
    results = search_client.search(search_text=current_ground_truth, top=20, query_type=QueryType.SEMANTIC, semantic_configuration_name=semantic_configuration_name)

    # Dictionary to hold the search results with an identifier
    search_results = {}

    # Process and display the results
    identifier = 1  # Start identifier at 1
    for result in results:
        # Remove newline characters and store results in dictionary
        content = result['content'].replace('\n', ' ')
        metadata_storage_name = result['metadata_storage_name']
        search_results[identifier] = {
            'content': content,
            'metadata_storage_name': metadata_storage_name
        }

        # Print results with identifier
        print(f"{identifier}. Content: {content}")
        print(f"   Metadata Storage Name: {metadata_storage_name}\n")

        identifier += 1  # Increment identifier for the next result

    return search_results

evaluator_client = AzureOpenAI(
    azure_endpoint = openai_gpt4_api_base,
    api_key = openai_gpt4_api_key,
    api_version = openai_api_version
)

def evaluate_response(current_ground_truth, ground_truth_sources):
    system_prompt = f"""You are a Ground Truth Evaluator, tasked with ensuring that information remains accurate and up-to-date. Given that ground truth can evolve over time, your role involves revising the existing ground truth to align with the latest validated information. Below is the current ground truth followed by a list of potential updates. Please review and select the most accurate and recent update from the list provided.

**Example Caution:** Be vigilant and avoid confusion between similar sounding product names like "Produk Simpanan Deposito Valas" and "Produk Simpanan Deposito Rupiah" which, despite their similarity, refer to different products.

**Current Ground Truth:**
{current_ground_truth}

**Latest Ground Truth Sources:**
{ground_truth_sources}

Please respond in JSON format, specifying the number of the selected ground truth source. Use the format: `{{"selected_source_number": }}`
"""

    # Creating List of Role Messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Creating Chat Completion for Evaluator Bot
    completion = evaluator_client.chat.completions.create(
        model=openai_gpt4_deployment_name,
        response_format={ "type": "json_object" },
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=100,
        seed=42,
    )

    evaluator_response = completion.choices[0].message.content

    json_response = json.loads(evaluator_response)

    return json_response

def judge_response(current_ground_truth, ground_truth_sources, revised_ground_truth):
    system_prompt = f"""As a Ground Truth Reviewer, your expertise lies in meticulously ensuring that updates to information align accurately with the most recent and relevant sources. Given that ground truth can change over time, you are provided with the original ground truth, a revised version proposed by your team, and a list of the latest ground truth sources. Your task is to assess whether the revised ground truth aligns appropriately with the best and most current information available. If you find another source from the list that more accurately reflects the needed updates than the revised version your team proposed, you should determine the revision as inadequate.

**Example Caution 1:** Be vigilant and avoid confusion between similar sounding product names like "Produk Simpanan Deposito Valas" and "Produk Simpanan Deposito Rupiah" which, despite their similarity, refer to different products.

**Example Caution 2:** Carefully distinguish between "Produk Simpanan Deposito BRImo" and "Produk Simpanan Deposito Rupiah," as these are distinct products with potentially different features and terms.

**Initial Ground Truth:**
{current_ground_truth}

**Latest Ground Truth Sources:**
{ground_truth_sources}

**Proposed Revised Ground Truth:**
{revised_ground_truth}

Assess the suitability of the revised ground truth and respond in JSON format, indicating whether the revision passes your evaluation. Use the format: `{{"pass": 'true/false'}}`"""

    # Creating List of Role Messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Creating Chat Completion for Evaluator Bot
    completion = evaluator_client.chat.completions.create(
        model=openai_gpt4_deployment_name,
        response_format={ "type": "json_object" },
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=100,
        seed=42,
    )

    evaluator_response = completion.choices[0].message.content

    json_response = json.loads(evaluator_response)

    return json_response

# Example usage
if __name__ == "__main__":
    dotenv.load_dotenv()

    current_ground_truth = """Deskripsi Produk Pinjaman Kredit Agunan Kas: Kredit Agunan Kas adalah pinjaman yang seluruh agunannya berupa giro, deposito, atau setara kas lainnya"""
    results = search_documents(current_ground_truth=current_ground_truth)

    evaluation_count = 0  # Initialize the counter for evaluations

    while evaluation_count < 2 and results:
        evaluation_count += 1  # Increment evaluation counter at each loop

        # Evaluate the response and get the selected source number
        evaluator_output = evaluate_response(current_ground_truth, results)
        selected_source_number = evaluator_output.get('selected_source_number', None)

        # Check if the selected source number still exists in results
        if selected_source_number not in results:
            print("Selected source no longer exists. Adjusting selection.")
            selected_source_number = next(iter(results), None)
        
        if selected_source_number is None:
            print("No valid source to evaluate.")
            break

        # Print the result based on the selected source number
        selected_result = results[selected_source_number]
        print(f"Accessing result {selected_source_number}:")
        print(f"Content: {selected_result['content']}")
        print(f"Metadata Storage Name: {selected_result['metadata_storage_name']}")

        # Check if the selected result passes the ground truth check
        judge_output = judge_response(current_ground_truth, results, selected_result['content'])
        print(judge_output)  # Debug print to check the actual output
        pass_value = judge_output['pass']
        print(f"Pass value type: {type(pass_value)}, value: {pass_value}")  # Debugging the type and value

        # Normalize the pass_value to boolean if it's string
        if isinstance(pass_value, str):
            pass_value = pass_value.lower() == 'true'

        # Check the normalized boolean value
        if not pass_value:
            print(f"Removing result {selected_source_number} and re-evaluating")
            del results[selected_source_number]  # Remove the entry that didn't pass
        else:
            print("Selected result is valid.")
            break

    if evaluation_count >= 3:
        print("Maximum evaluation attempts reached. No more evaluations will be performed.")
