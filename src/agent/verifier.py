"""Autonomous fact-checking verification agent using LangChain, Tavily Search, and DuckDuckGo fallback."""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_VERIFICATION_PROMPT = (
    "You are an objective multimodal fact-checking assistant. "
    "An image-caption pair was flagged as potentially out-of-context (consistency score S_c={consistency_score:.2f}). "
    "Suspected misaligned caption entities/tokens: {misaligned_tokens}.\n\n"
    "Claim / Caption:\n\"{caption}\"\n\n"
    "Retrieved Web Evidence:\n{evidence_text}\n\n"
    "Task: Synthesize the retrieved evidence into a concise, factual 2-sentence debunking or verification "
    "summary referencing the retrieved sources. Do not fabricate facts outside the provided evidence."
)


class VerificationAgent:
    """Autonomous RAG verification agent invoked when S_c < triage_threshold.

    Search pipeline:
        1. Primary: Tavily Search API (`tavily-python`)
        2. Fallback: DuckDuckGo Search (`ddgs` / `duckduckgo-search`) on missing key,
           package absence, quota exhaustion, or network error.
        3. Graceful offline fallback if neither search tool succeeds.

    Synthesis pipeline:
        1. Primary: LangChain + Gemini (`langchain-google-genai`) or Ollama (`langchain-community`).
        2. Fallback: Extractive 2-sentence summary built directly from retrieved search snippets.
    """

    def __init__(
        self,
        search_engine: str = "tavily",
        fallback_search_engine: str = "duckduckgo",
        max_search_results: int = 3,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-1.5-flash",
        tavily_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.search_engine = (search_engine or "tavily").lower()
        self.fallback_search_engine = (fallback_search_engine or "duckduckgo").lower()
        self.max_search_results = max_search_results
        self.llm_provider = (llm_provider or "gemini").lower()
        self.llm_model = llm_model
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

    def build_search_query(
        self,
        caption: str,
        misaligned_tokens: Optional[List[str]] = None,
    ) -> str:
        """Constructs a concise fact-checking query from caption and misaligned tokens."""
        clean_caption = re.sub(r"\s+", " ", caption).strip()
        # Keep query concise for search APIs
        words = clean_caption.split()
        core_caption = " ".join(words[:18])

        extra_terms = ""
        if misaligned_tokens:
            unique_tokens = []
            seen = set()
            caption_lower = core_caption.lower()
            for tok in misaligned_tokens:
                cleaned = tok.strip()
                if cleaned and cleaned.lower() not in seen and cleaned.lower() not in caption_lower:
                    seen.add(cleaned.lower())
                    unique_tokens.append(cleaned)
            if unique_tokens:
                extra_terms = " " + " ".join(unique_tokens[:3])

        return f"{core_caption}{extra_terms} fact check".strip()

    def _search_tavily(self, query: str) -> List[Dict[str, str]]:
        """Executes search via Tavily API. Raises on missing credentials or failure."""
        api_key = self.tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set.")

        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=self.max_search_results,
            search_depth="basic",
        )
        raw_results = response.get("results", []) if isinstance(response, dict) else []
        parsed: List[Dict[str, str]] = []
        for item in raw_results[: self.max_search_results]:
            parsed.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "content": str(item.get("content", "")).strip(),
                }
            )
        return parsed

    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Executes search via DuckDuckGo (`ddgs` or `duckduckgo_search`)."""
        ddgs_cls = None
        try:
            from ddgs import DDGS  # type: ignore

            ddgs_cls = DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore

            ddgs_cls = DDGS

        parsed: List[Dict[str, str]] = []
        with ddgs_cls() as ddgs:
            raw_results = list(ddgs.text(query, max_results=self.max_search_results))

        for item in raw_results[: self.max_search_results]:
            parsed.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("href") or item.get("url") or "").strip(),
                    "content": str(item.get("body") or item.get("snippet") or "").strip(),
                }
            )
        return parsed

    def search_evidence(self, query: str) -> Tuple[List[Dict[str, str]], str, List[str]]:
        """Retrieves live search evidence with automatic Tavily -> DuckDuckGo fallback.

        Returns:
            Tuple of (evidence_list, provider_used, warning_messages).
        """
        warnings: List[str] = []

        if self.search_engine == "tavily":
            try:
                results = self._search_tavily(query)
                if results:
                    return results, "tavily", warnings
                warnings.append("Tavily returned empty results; falling back to DuckDuckGo.")
            except Exception as exc:
                warnings.append(f"Tavily search unavailable ({exc}); falling back to DuckDuckGo.")

        if self.fallback_search_engine == "duckduckgo" or self.search_engine == "duckduckgo":
            try:
                results = self._search_duckduckgo(query)
                if results:
                    return results, "duckduckgo", warnings
                warnings.append("DuckDuckGo returned empty results.")
            except Exception as exc:
                warnings.append(f"DuckDuckGo search unavailable ({exc}).")

        return [], "offline_fallback", warnings

    def _synthesize_with_langchain(
        self,
        caption: str,
        consistency_score: float,
        misaligned_tokens: List[str],
        evidence: List[Dict[str, str]],
    ) -> Optional[Tuple[str, str]]:
        """Synthesizes a 2-sentence rationale via LangChain + Gemini/Ollama if configured.

        Returns:
            (rationale, llm_provider_used) or None if unavailable.
        """
        evidence_lines = []
        for idx, item in enumerate(evidence, start=1):
            snippet = item.get("content", "")[:350]
            evidence_lines.append(
                f"[{idx}] {item.get('title', 'Source')} ({item.get('url', 'N/A')}): {snippet}"
            )
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No external web snippets retrieved."

        prompt_text = DEFAULT_VERIFICATION_PROMPT.format(
            consistency_score=consistency_score,
            misaligned_tokens=", ".join(misaligned_tokens) if misaligned_tokens else "None isolated",
            caption=caption,
            evidence_text=evidence_text,
        )

        try:
            from langchain_core.messages import HumanMessage  # type: ignore

            llm = None
            provider_used = None

            if self.llm_provider == "gemini":
                api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

                    llm = ChatGoogleGenerativeAI(
                        model=self.llm_model,
                        google_api_key=api_key,
                        temperature=0.1,
                    )
                    provider_used = f"langchain-gemini ({self.llm_model})"
            elif self.llm_provider == "ollama":
                from langchain_community.chat_models import ChatOllama  # type: ignore

                llm = ChatOllama(model=self.llm_model, temperature=0.1)
                provider_used = f"langchain-ollama ({self.llm_model})"

            if llm is not None and provider_used is not None:
                response = llm.invoke([HumanMessage(content=prompt_text)])
                content = getattr(response, "content", str(response)).strip()
                if content:
                    return content, provider_used
        except Exception:
            return None

        return None

    def _synthesize_extractive_fallback(
        self,
        caption: str,
        consistency_score: float,
        misaligned_tokens: List[str],
        evidence: List[Dict[str, str]],
        warnings: List[str],
    ) -> Tuple[str, str]:
        """Generates a deterministic 2-sentence summary when LLM synthesis is unavailable."""
        token_note = (
            f" (weakly grounded tokens: {', '.join(misaligned_tokens[:4])})"
            if misaligned_tokens
            else ""
        )
        sentence_1 = (
            f"Cross-modal screening flagged this post as potentially out-of-context "
            f"(consistency score S_c = {consistency_score:.2f}{token_note})."
        )

        if evidence:
            top = evidence[0]
            snippet = re.sub(r"\s+", " ", top.get("content", "")).strip()
            snippet = snippet[:200].rstrip(" ,.;") + ("..." if len(snippet) > 200 else "")
            source_title = top.get("title") or top.get("url") or "retrieved web source"
            sentence_2 = f"Retrieved evidence from {source_title} reports: \"{snippet}\""
        else:
            reason = warnings[0] if warnings else "External search/LLM credentials are not configured."
            sentence_2 = (
                f"Live web verification could not retrieve external snippets in this environment ({reason})"
            )

        return f"{sentence_1} {sentence_2}", "extractive_fallback"

    def verify(
        self,
        caption: str,
        consistency_score: float,
        misaligned_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs live search + evidence synthesis for a flagged image-caption pair."""
        start_time = time.perf_counter()
        tokens = misaligned_tokens or []

        query = self.build_search_query(caption=caption, misaligned_tokens=tokens)
        evidence, search_provider, warnings = self.search_evidence(query)

        llm_result = None
        if evidence:
            llm_result = self._synthesize_with_langchain(
                caption=caption,
                consistency_score=consistency_score,
                misaligned_tokens=tokens,
                evidence=evidence,
            )

        if llm_result is not None:
            rationale, llm_provider_used = llm_result
        else:
            rationale, llm_provider_used = self._synthesize_extractive_fallback(
                caption=caption,
                consistency_score=consistency_score,
                misaligned_tokens=tokens,
                evidence=evidence,
                warnings=warnings,
            )

        citations = [item["url"] for item in evidence if item.get("url")]
        agent_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "verdict": "Out-of-Context",
            "confidence_score": round(float(max(0.0, min(1.0, 1.0 - consistency_score))), 4),
            "rationale": rationale,
            "citations": citations,
            "evidence": evidence,
            "search_query": query,
            "search_provider": search_provider,
            "llm_provider_used": llm_provider_used,
            "warnings": warnings,
            "agent_latency_ms": round(agent_latency_ms, 2),
        }

