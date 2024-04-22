import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os, dotenv, json
from openai import AzureOpenAI
import signal
import sys

openai_api_type = os.getenv("AZURE_OPENAI_API_TYPE")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Azure OpenAI Setup GPT-4
openai_gpt4_api_base = os.getenv("AZURE_OPENAI_GPT4_API_BASE")
openai_gpt4_api_key = os.getenv("AZURE_OPENAI_GPT4_API_KEY")
openai_gpt4_deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")

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
    results = search_client.search(search_text=current_ground_truth, top=10)

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
        # print(f"{identifier}. Content: {content}")
        # print(f"   Metadata Storage Name: {metadata_storage_name}\n")

        identifier += 1  # Increment identifier for the next result

    return search_results

evaluator_client = AzureOpenAI(
    azure_endpoint = openai_gpt4_api_base,
    api_key = openai_gpt4_api_key,
    api_version = openai_api_version
)

def evaluate_response(current_ground_truth, ground_truth_sources):
    system_prompt = f"""You are a Ground Truth Evaluator, tasked with ensuring that information remains accurate and up-to-date. Given that ground truth can evolve over time, your role involves revising the existing ground truth to align with the latest validated information. Below is the current ground truth followed by a list of potential updates. Please review and select the most accurate and recent update from the list provided.

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
        max_tokens=1000,
        seed=42,
    )

    evaluator_response = completion.choices[0].message.content

    json_response = json.loads(evaluator_response)

    return json_response

def judge_response(current_ground_truth, ground_truth_sources, revised_ground_truth):
    system_prompt = f"""As a Ground Truth Reviewer, your expertise lies in meticulously ensuring that updates to information align accurately with the most recent and relevant sources. Given that ground truth can change over time, you are provided with the original ground truth, a revised version proposed by your team, and a list of the latest ground truth sources. Your task is to assess whether the revised ground truth aligns appropriately with the best and most current information available. If you find another source from the list that more accurately reflects the needed updates than the revised version your team proposed, you should determine the revision as inadequate.

**Example Caution:** Be vigilant and avoid confusion between similar sounding product names like "Produk Simpanan Deposito Valas" and "Produk Simpanan Deposito Rupiah" which, despite their similarity, refer to different products "Rupiah" and "Valas".

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
        max_tokens=1000,
        seed=42,
    )

    evaluator_response = completion.choices[0].message.content

    json_response = json.loads(evaluator_response)

    return json_response

def handle_exit(sig, frame):
    print("Program terminated, saving current progress...")
    global data
    data.to_csv(output_csv_path, index=False)
    sys.exit(0)

def update_csv_based_on_judgements(csv_path, output_csv_path):
    global data
    data = pd.read_csv(csv_path)
    
    # Set signal handler for graceful termination
    signal.signal(signal.SIGINT, handle_exit)

    try:
        # Iterate over each row in DataFrame
        for index, row in data.iterrows():
            current_ground_truth = row['ground_truth']
            results = search_documents(current_ground_truth=current_ground_truth)

            evaluation_count = 0
            update_success = False

            print(f"Processing row {index + 1} of {len(data)}")  # Report current progress

            while evaluation_count < 3 and results:
                evaluation_count += 1  # Increment evaluation counter at each loop

                evaluator_output = evaluate_response(current_ground_truth, results)
                selected_source_number = evaluator_output.get('selected_source_number', None)

                if selected_source_number not in results:
                    print("Selected source no longer exists. Adjusting selection.")
                    selected_source_number = next(iter(results), None)
                
                if selected_source_number is None:
                    print("No valid source to evaluate.")
                    break

                selected_result = results[selected_source_number]
                judge_output = judge_response(current_ground_truth, results, selected_result['content'])
                pass_value = judge_output['pass']

                # Normalize the pass_value to boolean if it's string
                if isinstance(pass_value, str):
                    pass_value = pass_value.lower() == 'true'

                if pass_value:
                    print("Selected result is valid.")
                    data.at[index, 'source_by_llm'] = selected_result['metadata_storage_name']
                    data.at[index, 'reground_truth_by_llm'] = selected_result['content']
                    update_success = True
                    break
                else:
                    print(f"Removing result {selected_source_number} and re-evaluating")
                    del results[selected_source_number]

            if not update_success:
                print("Failed to validate the result, setting to null.")
                data.at[index, 'source_by_llm'] = None
                data.at[index, 'reground_truth_by_llm'] = None

            print(f"Row {index + 1}: {'Passed' if update_success else 'Failed'}, Source: {data.at[index, 'source_by_llm']}")

            if evaluation_count >= 3 and not update_success:
                print(f"Maximum evaluation attempts reached. No more evaluations will be performed for row {index + 1}.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Write the modified DataFrame back to a new CSV to save the progress
        data.to_csv(output_csv_path, index=False)
        print("CSV has been updated based on the evaluation results.")

# Example usage
if __name__ == "__main__":
    dotenv.load_dotenv()
    original_csv_path = "./test100.csv"
    output_csv_path = "./test100_output.csv"
    update_csv_based_on_judgements(original_csv_path, output_csv_path)
