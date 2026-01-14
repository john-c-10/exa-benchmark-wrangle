import os
from typing import Any

import httpx

from .base import SearchResult, Searcher



class WrangleSearcher(Searcher):
    name = "wrangle"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://app.usewrangle.com/api/benchmarks/v1/search",
        processor: str = "base",
        source_policy: dict | None = None,
        evalsOff: bool = False,
    ):
        self.api_key = api_key or os.getenv("WRANGLE_INTERNAL_KEY")
        if not self.api_key:
            raise ValueError(
                "Wrangle API key required - set Wrangle_API_KEY or pass api_key"
            )

        self.base_url = base_url
        self.processor = processor
        self.source_policy = source_policy
        self.evalsOff = evalsOff
        self._client = httpx.AsyncClient(timeout=None)  # Timeout handled at benchmark level

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "maxResults": num_results,
            "query": query,
            "evaluations": not self.evalsOff,
            "minResults": num_results,
        }

        if self.source_policy:
            payload["source_policy"] = self.source_policy

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        response = await self._client.post(
            self.base_url,
            headers=headers,
            json=payload,
        )
        if response.status_code >= 400:
            print(f"API Error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()

        results = []
        raw_results = data.get("candidates", [])
        top10 = data.get("returnedTop10", False)
        # print(raw_results)
        for i, result in enumerate(raw_results):
            # print(result)
            if i >= num_results:
                break

            # Build comprehensive text from candidate profile
            candidate = result.get("candidate", {})
            
            parts = []

            if candidate.get("headline"):
                parts.append(f"Headline: {candidate.get('headline')}")

            if candidate.get("currentTitle"):
                parts.append(f"Current Role: {candidate.get('currentTitle')}")

            if candidate.get("currentCompany"):
                parts.append(f"Current Company: {candidate.get('currentCompany')}")

            if candidate.get("location"):
                parts.append(f"Location: {candidate.get('location')}")

            if candidate.get("summary"):
                parts.append(f"Summary: {candidate.get('summary')}")

            # Include recent experience for context
            experience = candidate.get("experience") or candidate.get("experienceList") or []
            if experience:
                exp_strs = []
                for exp in experience[:3]:  # Limit to recent 3
                    title = exp.get("title", "")
                    company = exp.get("company", "") or exp.get("companyName", "")
                    if title or company:
                        exp_strs.append(f"{title} at {company}".strip(" at"))
                if exp_strs:
                    parts.append(f"Experience: {'; '.join(exp_strs)}")

            # Include skills (abbreviated)
            skills = candidate.get("skills") or candidate.get("skillsArray") or []
            if skills:
                parts.append(f"Skills: {', '.join(skills[:10])}")

            text = "\n".join(parts)

            results.append(
                SearchResult(
                    url=candidate.get("linkedinUrl", ""),
                    title=candidate.get("title", ""),
                    text=text,
                    metadata={
                        "rank": i,
                        # "author": result.get("author"),
                        "author": "Wrangle"
                        # "published_date": result.get("published_date") or result.get("publishedDate"),
                    },
                )
            )

        return results, top10

    async def close(self):
        await self._client.aclose()

