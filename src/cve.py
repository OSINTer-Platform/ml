from collections import Counter
from datetime import datetime, timezone
from hashlib import md5
import logging
from typing import Any, Generator, TypedDict, cast
import requests
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from src.inference import query_openai


from . import config_options
from modules.objects import (
    FullCVE,
    BaseArticle,
    CVSS3,
    CVSS2,
    CVEReference,
)


logger = logging.getLogger("osinter")


class NVDRequestHeaders(TypedDict, total=False):
    startIndex: int
    resultsPerPage: int
    lastModStartDate: str
    lastModEndDate: str


def sort_articles_by_cves(articles: list[BaseArticle]) -> dict[str, list[BaseArticle]]:
    sorted: dict[str, list[BaseArticle]] = {}

    for article in articles:
        for tags in article.tags.interesting:
            if tags.name == "CVE's":
                for cve in tags.values:
                    if cve in sorted:
                        sorted[cve].append(article)
                    else:
                        sorted[cve] = [article]

    return sorted


def get_article_common_keywords(articles: list[BaseArticle]) -> list[str]:
    keyword_counter = Counter(
        [keyword for article in articles for keyword in article.tags.automatic]
    )
    return [word for word, count in keyword_counter.most_common() if count > 1][:10]


def query_nvd(
    start_date: datetime | None = None, start_index: int = 0, batch_size: int = 2000
) -> Generator[list[dict[str, Any]], None, None]:
    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def query(
        url: str, params: dict[str, int | str], headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        raw = requests.get(url, params=params, headers=headers)
        raw.raise_for_status()
        json = raw.json()
        logger.debug(
            f'Queried {json["resultsPerPage"]} vulnerabilities, {json["totalResults"] - (json["startIndex"] + json["resultsPerPage"])} left'
        )
        return cast(list[dict[str, Any]], json["vulnerabilities"])

    QUERY_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?"

    paramaters: NVDRequestHeaders = {
        "startIndex": start_index,
        "resultsPerPage": batch_size,
    }
    headers = {"content-type": "application/json"}

    if config_options.NVD_KEY:
        headers["apiKey"] = config_options.NVD_KEY

    if start_date:
        paramaters["lastModStartDate"] = start_date.isoformat()
        paramaters["lastModEndDate"] = datetime.now().isoformat()

    while True:
        logger.debug(
            f"Querying NVD for start index {paramaters['startIndex']} and batch size {batch_size} {'with' if config_options.NVD_KEY else 'without'} key"
        )
        results = query(QUERY_URL, cast(dict[str, str | int], paramaters), headers)
        yield results
        if len(results) < batch_size:
            break

        paramaters["startIndex"] += batch_size


def validate_cve(cve: dict[str, Any], articles: list[BaseArticle]) -> FullCVE:
    """For validating data comming from NVD"""

    def extract_primary_cvss(contents: list[dict[str, Any]]) -> dict[str, Any]:
        for content in contents:
            if content["type"] == "Primary":
                return content

        return contents[0]

    def parse_datetime(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").replace(
            tzinfo=timezone.utc
        )

    name = cve["id"]
    description = ""

    for desc in cve["descriptions"]:
        if desc["lang"] == "en":
            description = desc["value"]
            break

    weaknesses = []

    if "weaknesses" in cve:
        for source in cve["weaknesses"]:
            for desc in source["description"]:
                if desc["lang"] == "en":
                    weaknesses.append(desc["value"])

    cvss2: None | CVSS2 = None
    cvss3: None | CVSS3 = None

    if "metrics" in cve:
        if "cvssMetricV2" in cve["metrics"]:
            cvss2 = CVSS2.model_validate(
                extract_primary_cvss(cve["metrics"]["cvssMetricV2"])
            )

        if "cvssMetricV31" in cve["metrics"]:
            cvss3 = CVSS3.model_validate(
                extract_primary_cvss(cve["metrics"]["cvssMetricV31"])
            )
        elif "cvssMetricV30" in cve["metrics"]:
            cvss3 = CVSS3.model_validate(
                extract_primary_cvss(cve["metrics"]["cvssMetricV30"])
            )

    references = [
        CVEReference.model_validate(reference) for reference in cve["references"]
    ]

    return FullCVE(
        id=md5(name.encode("utf-8")).hexdigest(),
        cve=name,
        title="",
        description=description,
        keywords=get_article_common_keywords(articles),
        publish_date=parse_datetime(cve["published"]),
        modified_date=parse_datetime(cve["lastModified"]),
        weaknesses=weaknesses,
        status=cve["vulnStatus"],
        documents={article.id for article in articles},
        dating={article.publish_date for article in articles},
        references=references,
        cvss2=cvss2,
        cvss3=cvss3,
    )


def generate_cve_title(cve: FullCVE) -> FullCVE:
    def generate_title(desc: str) -> str:
        try:
            prompt = """Summarize the following description of a cyber security vulnerability into a title of at most 10 words. The language used should be highly technical and in the following format:
[vulnerability type] in [product name] [consequence if descriped]
Only add the consequence if explicitly described in the description
Vulnerability description:
"""
            return query_openai([{"role": "user", "content": prompt + desc}])
        except:
            return ""

    cve.title = generate_title(cve.description)
    return cve
