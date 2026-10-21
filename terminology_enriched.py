"""
Enhanced Medical Terminology Lookup combining UMLS and DuckDuckGo web search
for enriched knowledge cards with contextual information.
Uses CAMEL SearchToolkit directly — no MCP subprocess required.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import logging
from typing import Dict, Any
from umls_toolkit import UMLSToolkit
from camel.toolkits import FunctionTool, SearchToolkit
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from prompts.variable_process import _terminology_knowledge_card_prompt, _knowledge_card_user_prompt

# Suppress verbose CAMEL logging
logging.getLogger("camel").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)




class EnrichedTerminologyLookup:
    """
    Combines UMLS toolkit with CAMEL SearchToolkit (DuckDuckGo) for enriched
    knowledge cards. No MCP subprocess — direct HTTP calls only.
    """

    def __init__(self, config_path: str = "config/web_browser.json"):
        self.config_path = config_path

    def umls_definiton_tool(self, term: str) -> Dict[str, Any]:
        """
        Look up a medical term in UMLS to retrieve its CUI, semantic types,
        and definitions from the SNOMED CT US vocabulary.

        Args:
            term (str): The medical term to look up (e.g. 'hypertension', 'urinary tract infection').

        Returns:
            Dict mapping each CUI found to its UMLS information including definitions and semantic types.
        """
        umls_results = {}
        umls_toolkit = UMLSToolkit()
        umls_searched_ui = umls_toolkit.search_terms(term)
        for cui in umls_searched_ui:
            umls_results[cui] = {}
            umls_results[cui]["search_term"] = term
            umls_infomation = umls_toolkit.get_cui_info(cui, sabs="SNOMEDCT_US", language="ENG")
            umls_results[cui].update(umls_infomation)
        return umls_results


    async def lookup_term(
        self,
        term: str,
        agent: ChatAgent
    ) -> Dict[str, Any]:
        """
        Lookup a term using UMLS and DuckDuckGo web search.

        Args:
            term: Medical term to search
            agent: CAMEL agent with UMLS and SearchToolkit tools

        Returns:
            Dict with 'term' and 'knowledge card' keys
        """
        print(f" 🔍 search: {term}")

        try:
            response = await agent.astep(_knowledge_card_user_prompt(term))
            result_text = response.msgs[0].content if response.msgs else "No results"

            if isinstance(result_text, str):
                try:
                    result_dict = json.loads(result_text)
                    knowledge_card = result_dict.get("knowledge card", result_text)
                except json.JSONDecodeError:
                    knowledge_card = result_text
            else:
                knowledge_card = result_text.get("knowledge card", str(result_text))

            return {
                'term': term,
                'knowledge card': knowledge_card,
            }

        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            return {
                'term': term,
                'knowledge card': f"Error: {str(e)}",
            }

    async def _worker(
        self,
        worker_id: int,
        terms: list,
        agent: ChatAgent
    ) -> list:
        results = []
        for idx, term in enumerate(terms):
            print(f"[Worker {worker_id}] Processing {idx + 1}/{len(terms)}: {term}")
            result = await self.lookup_term(term, agent)
            results.append(result)
            await asyncio.sleep(0.2)
        return results


    async def process_raw_variable_enriched(
        self,
        input_json: str = "extracted_variables_raw.json",
        output_json: str = "knowledge_card.json",
        num_workers: int = 5
    ) -> str:
        """
        Process raw variables JSON with enriched lookup using parallel workers.

        Args:
            input_json: Path to input JSON (list of dicts with 'variable' key)
            output_json: Path to save enriched knowledge cards as JSON
            num_workers: Number of parallel workers (default: 5)

        Returns:
            Path to the output JSON file
        """
        if not os.path.exists(input_json):
            raise FileNotFoundError(f"Input JSON not found: {input_json}")

        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        terms = list({
            entry.get('variable') or entry.get('term') or entry.get('name')
            for entry in data
            if entry.get('variable') or entry.get('term') or entry.get('name')
        })
        print(f"\n📋 Loaded {len(terms)} unique terms from {input_json}")

        print(f"\n🔌 Setting up DuckDuckGo + UMLS tools...")

        umls_tool = FunctionTool(self.umls_definiton_tool)
        ddg_tool = FunctionTool(SearchToolkit().search_duckduckgo)
        all_tools = [umls_tool, ddg_tool]

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-5.4",
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={"temperature": 0.2, "max_tokens": 100000},
        )

        agents = [
            ChatAgent(model=model, system_message=_terminology_knowledge_card_prompt(), tools=all_tools)
            for _ in range(num_workers)
        ]

        terms_per_worker = max(1, len(terms) // num_workers)
        worker_tasks = []

        print(f"🚀 Starting {num_workers} workers in parallel...")
        for i in range(num_workers):
            start_idx = i * terms_per_worker
            end_idx = start_idx + terms_per_worker if i < num_workers - 1 else len(terms)
            worker_terms = terms[start_idx:end_idx]
            if not worker_terms:
                continue
            print(f"  Worker {i + 1}: {len(worker_terms)} terms")
            worker_tasks.append(self._worker(i + 1, worker_terms, agents[i]))

        worker_results = await asyncio.gather(*worker_tasks)

        results = [item for batch in worker_results for item in batch]

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"💾 Saved {len(results)} enriched terms to: {output_json}")
        print(f"{'=' * 60}")

        return output_json


async def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enriched medical terminology standardization (UMLS + DuckDuckGo)"
    )
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--term", help="Single term to lookup")
    parser.add_argument("--workers", type=int, default=5)

    args = parser.parse_args()

    lookup = EnrichedTerminologyLookup()

    if args.term:
        umls_tool = FunctionTool(lookup.umls_definiton_tool)
        ddg_tool = FunctionTool(SearchToolkit().search_duckduckgo)
        model=ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type="gpt-5.4",
                url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                model_config_dict={"temperature": 0.2, "max_completion_tokens":100000},
            )
        agent = ChatAgent(model=model, system_message=_terminology_knowledge_card_prompt(), tools=[umls_tool, ddg_tool])
        result = await lookup.lookup_term(args.term, agent)
        print("\n" + "=" * 60)
        print("ENRICHED RESULTS:")
        print("=" * 60)
        print(json.dumps(result, indent=2))
    else:
        await lookup.process_raw_variable_enriched(
            input_json=args.input,
            output_json=args.output,
            num_workers=args.workers
        )


if __name__ == "__main__":
    asyncio.run(main())
