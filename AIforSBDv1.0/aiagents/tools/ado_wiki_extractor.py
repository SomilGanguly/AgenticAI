import requests
import urllib.parse
import base64
import re
import uuid

def get_pat_header(pat_token):
    pat_token = pat_token.replace('"', '')  # Remove quotes if present
    pat_bytes = f":{pat_token}".encode("utf-8")
    pat_base64 = base64.b64encode(pat_bytes).decode("utf-8")
    return {"Authorization": f"Basic {pat_base64}"}

def extract_wiki_and_page_from_url(project_url, pat_token):
    """
    Returns (organization, project, wiki_name_or_id, page_path)
    page_path will be '/' for the root wiki page, or the path to the subpage.
    Handles both path-based and ID-based wiki URLs.
    Decodes percent-encoded path segments.
    """
    try:
        parts = urllib.parse.urlparse(project_url)
        # Decode each path segment
        path_parts = [urllib.parse.unquote(p) for p in parts.path.strip("/").split("/")]
        organization = path_parts[0]
        project = path_parts[1]
        wiki_name_or_id = None
        page_path = "/"
        # Find wiki name/id and page path
        if "_wiki" in path_parts and "wikis" in path_parts:
            wiki_idx = path_parts.index("wikis")
            wiki_name_or_id = path_parts[wiki_idx + 1]
            # Handle numeric page ID in URL (e.g., .../wikis/Migration Wiki/110/Design-Patterns)
            if len(path_parts) > wiki_idx + 2 and path_parts[wiki_idx + 2].isdigit():
                page_id = path_parts[wiki_idx + 2]
                page_path = get_page_path_from_id(organization, project, wiki_name_or_id, page_id, pat_token)
            elif "pages" in path_parts:
                pages_idx = path_parts.index("pages")
                if len(path_parts) > pages_idx + 1:
                    page_path = "/" + "/".join(path_parts[pages_idx + 1:])
            elif len(path_parts) > wiki_idx + 2:
                # Path-based URL after wiki name
                page_path = "/" + "/".join(path_parts[wiki_idx + 2:])
        if not wiki_name_or_id:
            wikis = get_wiki_list(organization, project, pat_token)
            if wikis:
                wiki_name_or_id = wikis[0].get("name") or wikis[0].get("id")
        return organization, project, wiki_name_or_id, page_path
    except Exception:
        raise ValueError("Could not parse organization/project/wiki/page from URL.")

def get_page_path_from_id(organization, project, wiki_name, page_id, pat_token):
    """
    Given a page ID, fetch the page metadata to get its path.
    """
    wiki_name_encoded = urllib.parse.quote(wiki_name, safe='')
    url = f"https://dev.azure.com/{organization}/{project}/_apis/wiki/wikis/{wiki_name_encoded}/pages/{page_id}?api-version=7.1"
    headers = get_pat_header(pat_token)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("path", "/")

def get_wiki_list(organization, project, pat_token):
    url = f"https://dev.azure.com/{organization}/{project}/_apis/wiki/wikis?api-version=7.1"
    headers = get_pat_header(pat_token)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("value", [])

def get_wiki_pages_tree(organization, project, wiki_name, pat_token):
    wiki_name_encoded = urllib.parse.quote(wiki_name, safe='')
    url = f"https://dev.azure.com/{organization}/{project}/_apis/wiki/wikis/{wiki_name_encoded}/pages?includeContent=false&recursionLevel=full&api-version=7.1"
    pat_bytes = f":{pat_token}".encode("utf-8")
    pat_base64 = base64.b64encode(pat_bytes).decode("utf-8")
    headers = {"Authorization": f"Basic {pat_base64}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def find_subtree_for_path(tree, target_path):
    """
    Recursively find the subtree (page and its subPages) for the given path.
    """
    if tree.get("path") == target_path:
        return tree
    for subpage in tree.get("subPages", []):
        result = find_subtree_for_path(subpage, target_path)
        if result:
            return result
    return None

def collect_paths_from_tree(tree):
    """
    Recursively collect all page paths from the given subtree.
    """
    paths = []
    if "path" in tree:
        paths.append(tree["path"])
    for subpage in tree.get("subPages", []):
        paths.extend(collect_paths_from_tree(subpage))
    return paths

def get_wiki_repo_id(organization, project, wiki_name, pat_token):
    url = f"https://dev.azure.com/{organization}/{project}/_apis/wiki/wikis/{wiki_name}?api-version=7.1"
    pat_bytes = f":{pat_token}".encode("utf-8")
    pat_base64 = base64.b64encode(pat_bytes).decode("utf-8")
    headers = {"Authorization": f"Basic {pat_base64}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("repositoryId")

def get_all_attachments_paths(organization, project, repo_id, pat_token):
    """
    Returns a list of all .attachments folder paths in the repo.
    """
    url = f"https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo_id}/items?recursionLevel=Full&api-version=7.1"
    headers = get_pat_header(pat_token)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    items = resp.json().get("value", [])
    attachment_dirs = set()
    for item in items:
        if item.get("isFolder") and ".attachments" in item.get("path", ""):
            attachment_dirs.add(item["path"])
    return list(attachment_dirs)

def try_download_image_from_any_attachments(img_filename, organization, project, repo_id, pat_token):
    """
    Try to download the image from any .attachments folder in the repo.
    """
    attachment_dirs = get_all_attachments_paths(organization, project, repo_id, pat_token)
    headers = get_pat_header(pat_token)
    for dir_path in attachment_dirs:
        candidate_path = f"{dir_path}/{img_filename}"
        candidate_path_encoded = urllib.parse.quote(candidate_path)
        git_items_url = (
            f"https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo_id}/Items"
            f"?path={candidate_path_encoded}&download=false&resolveLfs=true&%24format=octetStream&api-version=5.0-preview.1"
        )
        try:
            img_resp = requests.get(git_items_url, headers=headers)
            if img_resp.status_code == 200:
                return img_resp.content
        except Exception:
            continue
    return None

def extract_text_and_images(page_path, organization, project, wiki_name, pat_token):
    wiki_name_encoded = urllib.parse.quote(wiki_name, safe='')
    path_encoded = urllib.parse.quote(page_path, safe='')
    url = f"https://dev.azure.com/{organization}/{project}/_apis/wiki/wikis/{wiki_name_encoded}/pages?path={path_encoded}&includeContent=true&api-version=7.1"
    pat_bytes = f":{pat_token}".encode("utf-8")
    pat_base64 = base64.b64encode(pat_bytes).decode("utf-8")
    headers = {"Authorization": f"Basic {pat_base64}"}
    print(f"[DEBUG] Fetching wiki page content: {url}")
    response = requests.get(url, headers=headers)
    print(f"[DEBUG] Raw content response for path {page_path}: {response.text[:500]}...")
    response.raise_for_status()
    data = response.json()
    text = data.get("content", "")
    print(f"[DEBUG] Extracted text (first 500 chars): {text[:500]}")

    # Regex to find all markdown image links: ![alt](url)
    image_pattern = r'!\[.*?\]\((.*?)\)'
    image_links = re.findall(image_pattern, text)
    print(f"[DEBUG] Image links found: {image_links}")
    images = []

    repo_id = get_wiki_repo_id(organization, project, wiki_name, pat_token)
    print(f"[DEBUG] Wiki repo_id: {repo_id}")

    for img_url in image_links:
        img_content = None
        if img_url.startswith("/.attachments/"):
            filename = img_url.split("/")[-1]
            # Try default path first
            if '%' in img_url:
                attachment_path = img_url
            else:
                attachment_path = urllib.parse.quote(img_url, safe='')
            git_items_url = (
                f"https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo_id}/Items"
                f"?path={attachment_path}&download=false&resolveLfs=true&%24format=octetStream&api-version=5.0-preview.1"
            )
            print(f"[DEBUG] Attempting to download image from: {git_items_url}")
            try:
                img_resp = requests.get(git_items_url, headers=headers)
                print(f"[DEBUG] Image download status: {img_resp.status_code}, content length: {len(img_resp.content)}")
                img_resp.raise_for_status()
                img_content = img_resp.content
            except Exception as e:
                print(f"[ERROR] Failed to download image {img_url} from default path: {e}")
                # Try all .attachments folders in the repo
                img_content = try_download_image_from_any_attachments(filename, organization, project, repo_id, pat_token)
                if img_content:
                    print(f"[DEBUG] Successfully found image {filename} in another .attachments folder.")
                else:
                    print(f"[ERROR] Could not find image {filename} in any .attachments folder.")
        else:
            # External or absolute URL
            filename = img_url.split("/")[-1]
            git_items_url = img_url
            print(f"[DEBUG] Attempting to download image from: {git_items_url}")
            try:
                img_resp = requests.get(git_items_url, headers=headers)
                print(f"[DEBUG] Image download status: {img_resp.status_code}, content length: {len(img_resp.content)}")
                img_resp.raise_for_status()
                img_content = img_resp.content
            except Exception as e:
                print(f"[ERROR] Failed to download image {img_url}: {e}")

        if img_content:
            img_b64 = base64.b64encode(img_content).decode("utf-8")
            image_id = str(uuid.uuid4())
            images.append({
                "id": image_id,
                "filename": filename,
                "url": img_url,
                "base64": img_b64
            })

    return text, images

def extract_wiki_subtree_content(project_url, pat_token):
    organization, project, wiki_name_or_id, page_path = extract_wiki_and_page_from_url(project_url, pat_token)
    print(f"[INFO] Extracting from org: {organization}, project: {project}, wiki: {wiki_name_or_id}, page: {page_path}")
    tree = get_wiki_pages_tree(organization, project, wiki_name_or_id, pat_token)
    subtree = find_subtree_for_path(tree, page_path)
    if not subtree:
        print(f"[ERROR] Could not find page path {page_path} in wiki {wiki_name_or_id}")
        return "", []
    page_paths = collect_paths_from_tree(subtree)
    print(f"[INFO] Found {len(page_paths)} pages under this subtree.")
    all_text = ""
    all_images = []
    for subpage_path in page_paths:
        text, images = extract_text_and_images(subpage_path, organization, project, wiki_name_or_id, pat_token)
        all_text += f"\n\n---\n\n# Page: {subpage_path}\n\n{text}"
        all_images.extend(images)
    return all_text, all_images
