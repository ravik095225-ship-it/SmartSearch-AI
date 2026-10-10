from ddgs import DDGS
from urllib.parse import urlparse
import re


MAX_RESULTS = 8


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_domain(url):
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def is_result_relevant_to_query(query: str, title: str, body: str) -> bool:
    """
    Ensure the retrieved web result actually matches at least one meaningful token
    from the user query, avoiding DuckDuckGo returning unrelated fallbacks
    for gibberish or nonsensical queries.
    """
    raw_tokens = re.findall(r"\b\w+\b", query.lower())
    stopwords = {"who", "is", "the", "of", "in", "what", "a", "an", "and", "or", "for", "on", "at", "to", "by", "from", "how"}
    meaningful = [t for t in raw_tokens if len(t) > 2 and t not in stopwords]
    tokens_to_check = meaningful if meaningful else raw_tokens
    if not tokens_to_check:
        return True
    
    haystack = f"{title} {body}".lower()
    return any(t in haystack for t in tokens_to_check)


def search_web(query, max_results=MAX_RESULTS):
    """
    Search the public web and normalize results
    into SmartSearch's internal format.
    """

    query = query.strip()

    if not query:
        return []

    try:

        ddgs = DDGS(
            timeout=10
        )

        results = ddgs.text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=max_results,
            backend="auto"
        )

    except Exception as error:

        print("WEB SEARCH ERROR:", error)

        return []


    normalized = []


    for index, result in enumerate(results, start=1):

        title = clean_text(
            result.get("title", "")
        )

        url = result.get(
            "href",
            ""
        )

        body = clean_text(
            result.get("body", "")
        )


        if not title or not url:
            continue

        if not is_result_relevant_to_query(query, title, body):
            continue


        normalized.append({

            "id": index,

            "title": title,

            "url": url,

            "domain": get_domain(url),

            "source": get_domain(url),

            "filename": get_domain(url),

            "category": "Web",

            "snippet": body[:220] if len(body) > 220 else body,

            "content": body,

            "source_type": "web",

            "is_web": True,

            "score": round(max(2.0, 8.5 - (index * 0.6)), 2),

            "quality_score": 75,

            "spam_score": 0,

            "is_duplicate": False,

            "is_low_quality": False

        })


    return normalized


def test_web_search():

    print()
    print("=" * 60)
    print("SMARTSEARCH AI — WEB SEARCH TEST")
    print("=" * 60)
    print()

    query = input(
        "Enter web search query: "
    ).strip()

    if not query:

        print(
            "Please enter a query."
        )

        return


    print()
    print("Searching the web...")
    print()


    results = search_web(
        query,
        max_results=5
    )


    if not results:

        print(
            "No web results found."
        )

        return


    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}.",
            result["title"]
        )

        print(
            "   Domain:",
            result["domain"]
        )

        print(
            "   URL:",
            result["url"]
        )

        print(
            "   Snippet:",
            result["content"][:250]
        )

        print()


if __name__ == "__main__":

    try:

        test_web_search()

    except KeyboardInterrupt:

        print()
        print("Test stopped.")

    except Exception as error:

        print()
        print("ERROR:")
        print(error)