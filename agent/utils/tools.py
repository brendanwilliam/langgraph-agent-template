import os
from tavily import TavilyClient
from langchain_core.tools import tool
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from dotenv import load_dotenv
import requests
import re

# Load environment variables
load_dotenv()

@tool(response_format="content_and_artifact")
def tavily_search_and_extract(query: str, max_results: int = 3):
    """
    Get Tavily query for user input.

    Args:
        query: The query string.
        max_results: The maximum number of results to return.

    Returns:
        search, serialized_extract
    """
    try:
        # Get search results from Tavily
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        search = tavily_client.search(
            query=query,
            max_results=max_results,
            include_answer="basic"
        )

        # Get extracted information from Tavily
        urls = [result['url'] for result in search['results']]
        extract = tavily_client.extract(urls=urls)
        # Serialize the extracted results
        serialized_extract = "\n\n".join(
            f"Source: {doc['url']}\nContent: {doc['raw_content']}"
            for doc in extract['results']
        )

        # Create a tool message that summarizes the search
        tool_message = f"Searched for '{query}' and found {len(urls)} relevant results."

        return tool_message, serialized_extract

    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error during search: {str(e)}"
        return error_message, f"Failed to retrieve information about '{query}'. Error: {str(e)}"


@tool(response_format="content_and_artifact")
def google_search_and_extract(query: str, num_results: int = 1):
    """
    Perform a custom Google Search for the given query and extract the results into Markdown format.

    Args:
        query: The search query
        num_results: The number of results to return

    Returns:
        search, serialized_extract
    """

    # Function to fetch content from a URL
    def extract(url: str) -> str:
        """
        Fetch the content of a webpage and convert it into Markdown.

        Args:
            url: The URL to fetch

        Returns:
            str: The content of the webpage in Markdown format with source attribution
        """
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.extract()

            # Convert to markdown
            markdown = md(
                str(soup),
                heading_style="ATX",
                convert=["a", "p", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "ul", "ol", "li", "table"],
                escape_asterisks=False,
                escape_underscores=False
            )
            markdown = f"Source: {url}\n\nContent: {markdown}\n\n"
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)

            return markdown[:10000]

        except Exception as e:
            print(f"Error: {e}")
            return ""

    try:
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': os.getenv("GOOGLE_SEARCH_API_KEY"),
            'cx': os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            'q': query,
            'num': min(num_results, 5)  # API limit is 10 per request
        }

        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return [], ""

        results = response.json()['items']
        urls = [item['link'] for item in results]

        # Create markdown links with titles
        markdown_links = []
        for item in results:
            title = item.get('title', 'No Title')
            url = item.get('link', '')
            markdown_links.append(f"[{title}]({url})")

        extract = "\n\n".join(
            f"Source: {url}\nContent: {extract(url)}"
            for url in urls
        )

        # Create a tool message that summarizes the search
        tool_message = f"Searched for '{query}' and found {len(urls)} relevant sources:\n\n" + "\n".join(markdown_links)
        return tool_message, extract

    except Exception as e:
        print(f"Error: {e}")
        return "", ""