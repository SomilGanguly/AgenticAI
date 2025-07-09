import csv
import os
import subprocess

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from ruamel.yaml import YAML
import jsonpatch

KUBERNETES_CONFIGURATION_FILE = ".\\Data\\configuration_parameters.csv"

def write_to_file_md(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def load_instructions(filename):
    file_path = "./INSTRUCTIONS/" + filename
    if filename == "":
        print(f"File {file_path} does not exist.")
        return None
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()

def get_kv_secret_value(credential, secret_names):
    secrets = {}
    try:
        KEYVAULT_URL = os.environ[
            "KEYVAULT_URL"
        ]  # Must be set in environment or Azure App Settings
        secret_client = SecretClient(vault_url=KEYVAULT_URL, credential=credential)
        for secret_name in secret_names:
            secret_value = secret_client.get_secret(secret_name).value
            secrets[secret_name] = secret_value
        return secrets
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving secret: {e.stderr}")
        raise

def get_dict_by_criteria(dict_list, criteria):
    """
    Returns the first dictionary from dict_list that matches all key-value pairs in criteria.
    """
    if not dict_list or not criteria:
        return None
    
    # Iterate through each dictionary in the list
    for d in dict_list:
        if all(d.get(k) == v for k, v in criteria.items()):
            return d
    return None

def extract_jsonpaths(data, values_to_update, keys, old_values, path="", patch=None):
    paths = []
    if isinstance(data, dict):
        for k, v in data.items():
            k = k.replace("/", "~1") # Needed for JSON Patch compliance
            # if k == "creationTimestamp":
            #     print("Found it!")
            #     print(old_values)
            #     print(v == old_values)
            #     print(type(old_values[0]).__name__)
            #     print(k in keys, v in old_values)
            # print(f"Processing key: {k}, value: {v}")
            # print("Value Type:", type(k).__name__, type(v).__name__)
              
            new_path = f"{path}/{k}" if path else f"/{k}"
            paths.append(new_path)
            if k in keys and v in old_values:
                updated_value = get_dict_by_criteria(values_to_update, {"key": k, "old_value": v})
                # print(f"Updated value found: {updated_value}")
                patch_dict = {
                    "op": updated_value["action"],
                    "path": new_path,
                    "value": updated_value["new_value"]
                }
                patch.append(patch_dict)
            # print(f"New path added: {new_path}")
            # print("Extrating paths from value:", v)
            paths.extend(extract_jsonpaths(v, values_to_update, keys, old_values, new_path, patch)) 

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{path}/{idx}"
            paths.append(new_path)
            paths.extend(extract_jsonpaths(item, values_to_update, keys, old_values, new_path, patch))

    return paths, patch

def update_yaml_key(resource_type, data, values_to_update, destination_path):
    keys = [item["key"] for item in values_to_update]
    old_values = [item["old_value"] for item in values_to_update]
    # print("keys:", keys)
    # print("old values:", old_values)
    print("")
    try:
        paths = []
        patch_operations = []
        paths, patch_operations = extract_jsonpaths(data, values_to_update, keys, old_values, paths, patch_operations)
        print("************************END OF FUNCTION **********************")
        print("")
        print("Patch operations:", patch_operations)
        if not patch_operations:
            print("No patch operations generated")
            write_to_file(destination_path, data)
            return data
        print(f"Generated {len(patch_operations)} patch operations:")
        for op in patch_operations:
            print(f"  {op}")
        patched = jsonpatch.apply_patch(data, patch_operations)
        print("Patch applied successfully")

        write_to_file(destination_path, patched)
        return patched
    except jsonpatch.InvalidJsonPatch as e:
        print(f"Invalid JSON patch: {e}")
        write_to_file(destination_path, data)
        return data
    except Exception as e:
        print(f"Error applying patch: {e}")
        write_to_file(destination_path, data)
        return data

def read_file_if_exists(filepath):
    yaml = YAML()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            print(f"File {filepath} exists.")
            data = yaml.load(file)
            print(f"YAML content loaded from {filepath}.")
        return data
    else:
        return None
    
def write_to_file(filepath, data):
    if isinstance(data, dict):
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        with open(filepath, "w") as file:
            yaml.dump(data, file)
    else:
        with open(filepath, "w") as file:
            file.write(data)
    return filepath
    # yaml = YAML()
    # with open(filepath, "w") as file:
    #     yaml.dump(data, file)
    #     print(f"Data written to {filepath} successfully.")
    # return filepath

def write_to_csv(filepath, data):
    """ Appends data to a CSV file at the specified filepath."""
    # Add option to write to a new file if it doesn't exist
    with open(filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
            print("Writing row to CSV:", row)
        file.close()
    print(f"Data written to {filepath} successfully.")
    return filepath

# TODO: function to read from CSV file and get data for a certain value (ACR name or path of an attribute in a K8 YAML file)
def read_from_csv(filepath, conditions={}):
    """
    Reads a CSV file into a pandas DataFrame and returns rows where the specified column satisfies the condition.
    :param filepath: Path to the CSV file.
    :param condition: A function that takes a value and returns True if the row should be included.
    :return: Filtered DataFrame.
    """
    df = pd.read_csv(filepath)
    filtered_df = df.copy()
    for column, value in conditions.items():
        filtered_df = filtered_df[filtered_df[column] == value]
    return filtered_df

def main():
    print("Starting YAML update...")
    data = read_file_if_exists("SourceManifests/deployment_reviews-v2.yaml")
    # print(type(data).__name__)
    values_to_update = [
    {
        'key': 'creationTimestamp',
        'old_value': '2025-06-10T14:00:06Z',
        'new_value': 'na',
        'action': 'remove'
    },
    {
        'key': 'status',
        'old_value': {
            'availableReplicas': 1,
            'conditions': [
                {
                    'lastTransitionTime': '2025-06-10T14:00:06Z',
                    'lastUpdateTime': '2025-06-10T14:00:21Z',
                    'message': 'ReplicaSet "reviews-v2-587ddf67d9" has successfully progressed.',
                    'reason': 'NewReplicaSetAvailable',
                    'status': 'True',
                    'type': 'Progressing'
                },
                {
                    'lastTransitionTime': '2025-06-22T13:44:17Z',
                    'lastUpdateTime': '2025-06-22T13:44:17Z',
                    'message': 'Deployment has minimum availability.',
                    'reason': 'MinimumReplicasAvailable',
                    'status': 'True',
                    'type': 'Available'
                }
            ],
            'observedGeneration': 1,
            'readyReplicas': 1,
            'replicas': 1,
            'updatedReplicas': 1
        },
        'new_value': 'na',
        'action': 'remove'
    },
    {
        'key': 'image',
        'old_value': 'docker.io/istio/examples-bookinfo-reviews-v2:1.16.4',
        'new_value': 'conmigcr.azurecr.io/examples-bookinfo-reviews-v2:1.16.4',
        'action': 'replace'
    },
    {
        'key': 'kubectl.kubernetes.io~1last-applied-configuration',
        'old_value': '{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{"provider":"kubernetes-sample-apps"},"labels":{"app":"reviews","version":"v2"},"name":"reviews-v2","namespace":"bookinfo"},"spec":{"replicas":1,"selector":{"matchLabels":{"app":"reviews","version":"v2"}},"template":{"metadata":{"annotations":{"provider":"kubernetes-sample-apps"},"labels":{"app":"reviews","version":"v2"}},"spec":{"containers":[{"env":[{"name":"LOG_DIR","value":"/tmp/logs"}],"image":"docker.io/istio/examples-bookinfo-reviews-v2:1.16.4","imagePullPolicy":"IfNotPresent","name":"reviews","ports":[{"containerPort":9080}],"securityContext":{"runAsUser":1000},"volumeMounts":[{"mountPath":"/tmp","name":"tmp"},{"mountPath":"/opt/ibm/wlp/output","name":"wlp-output"}]}],"serviceAccountName":"bookinfo-reviews","volumes":[{"emptyDir":{},"name":"wlp-output"},{"emptyDir":{},"name":"tmp"}]}}}}\n',
        'new_value': 'na',
        'action': 'remove'
    }
]

    keys = [item["key"] for item in values_to_update]
    print("Keys to update:", keys)
    # old_values = [item["old_value"] for item in values_to_update]

    destination_path = ".//TargetManifests//test2.yaml"
    update_yaml_key("deployment", data, values_to_update, destination_path)
    

if __name__ == "__main__":
    main()

# In JSON Patch (RFC 6902), the possible values for the op field are:

# "add": Add a value at the target location.
# "remove": Remove the value at the target location.
# "replace": Replace the value at the target location.
# "move": Move a value from one location to another.
# "copy": Copy a value from one location to another.
# "test": Test that a value at the target location matches a specified value.
# These are the standard operations supported by JSON Patch.



