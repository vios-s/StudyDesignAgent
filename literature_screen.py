"""
Literature Screening Module using CAMEL-AI framework
Implements PECO-based screening criteria generation and multi-level paper evaluation
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import pandas as pd
import logging
import textwrap
from typing import Dict, Any, List
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from utils import *
from prompts.abstract_screening_process import (
    _abstract_criteria_generation_prompt,
    _abstract_paper_screening_prompt,
    _persona_derivation_system_prompt,
    _persona_derivation_user_prompt,
    _fallback_reviewer_personas,
    _abstract_reviewer_task_prompt,
    _abstract_reviewer_system_prompt,
)

# Suppress verbose CAMEL logging
logging.getLogger("camel").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)



class LiteratureScreener:
    """
    PECO-based literature screening with multi-level evaluation
    """

    def __init__(
        self,
        exposure: str,
        outcome: str,
        population: str = None,
        comparison: str = None,
        include_threshold: float = 1.0,
        resource_folder: str = None,
    ):
        """
        Initialize screener with PECO elements

        Args:
            exposure: Main exposure
            outcome: Primary outcome of interest
            population: Target population (optional)
            comparison: Comparison/control (optional)
            include_threshold: Proportion of YES answers required to INCLUDE (default 1.0 = 100%)
            resource_folder: Optional path to folder containing full-text PDFs named by PMID
                             (e.g. /resource/24113040.pdf). If None, PDF check is skipped.
        """
        self.exposure = exposure
        self.outcome = outcome
        self.population = population or "any population"
        self.comparison = comparison or "any comparison"
        self.include_threshold = include_threshold
        self.resource_folder = resource_folder

        # Initialize model
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-5.4-nano",
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={"temperature": 0.2, "max_tokens": 10000},
        )

        self.content_criteria = []

    def _criteria_generation_prompt(self, num_criteria: int = 6) -> str:
        """Generate prompt for creating screening criteria"""
        return _abstract_criteria_generation_prompt(num_criteria)

    def generate_criteria(self) -> Dict[str, List[Dict]]:
        """
        Generate screening criteria using LLM

        Returns:
            Dictionary with content_criteria list (6 criteria for abstract screening)
        """
        print(f"\n{'=' * 60}")
        print(f"Generating screening criteria...")
        print(f"{'=' * 60}\n")

        agent = ChatAgent(
            system_message=self._criteria_generation_prompt(num_criteria=6),
            model=self.model
        )

        response = agent.step("Generate screening criteria")
        result_text = response.msgs[0].content.strip()

        # Parse JSON from response
        if result_text.startswith('```json'):
            result_text = result_text.split('```json')[1].split('```')[0].strip()
        elif result_text.startswith('```'):
            result_text = result_text.split('```')[1].split('```')[0].strip()

        try:
            criteria = json.loads(result_text)
            self.content_criteria = criteria.get("content_criteria", [])

            return criteria

        except json.JSONDecodeError as e:
            print(f"✗ Error parsing criteria: {e}")
            print(f"Raw response: {result_text[:500]}")
            raise

    def _paper_screening_prompt(
        self,
        paper_content: str,
        criteria_list: List[Dict],
    ) -> str:
        """
        Generate TrialMind-style screening prompt with structured evaluation

        Args:
            paper_content: Paper title or title+abstract
            criteria_list: List of criteria dicts with 'criterion' and 'rationale'
            screening_level: 'title' or 'abstract'
        """
        return _abstract_paper_screening_prompt(
            self.population, self.exposure, self.comparison, self.outcome, paper_content, criteria_list
        )
    
    def _validate_evaluations(
        self,
        evaluations: List[str],
        expected_count: int
    ) -> List[str]:
        """
        Validate and normalize evaluation results (TrialMind-style)

        Args:
            evaluations: List of evaluation responses
            expected_count: Expected number of evaluations

        Returns:
            Validated list of evaluations, padded with YES if needed
        """
        # Normalize to uppercase
        normalized = [str(e).upper().strip() for e in evaluations]

        # Replace invalid values with YES
        valid_values = {"YES", "NO", "YES"}
        normalized = [e if e in valid_values else "YES" for e in normalized]

        # Pad or truncate to expected count
        if len(normalized) < expected_count:
            # Pad with YES
            normalized.extend(["YES"] * (expected_count - len(normalized)))
        elif len(normalized) > expected_count:
            # Truncate
            normalized = normalized[:expected_count]

        return normalized

    

class DualReviewerScreener(LiteratureScreener):
    """
    Two independent reviewer agents screen each paper separately.
    Their personas are derived from the research question's most relevant domains.
    Decision rule: INCLUDE if at least one reviewer votes INCLUDE.
    """

    def _derive_personas(self, research_question: str) -> List[Dict[str, str]]:
        """
        Use LLM to identify the two most relevant clinical/research domains
        from the research question and write a persona for each.
        Returns a list of two dicts: {role_name, persona}.
        """
        agent = ChatAgent(
            system_message=_persona_derivation_system_prompt(),
            model=self.model,
        )
        response = agent.step(
            _persona_derivation_user_prompt(research_question)
        )
        raw = response.msgs[0].content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()
        try:
            personas = json.loads(raw)
            if isinstance(personas, list) and len(personas) >= 2:
                return personas[:2]
        except Exception:
            pass
        # Fallback
        return _fallback_reviewer_personas()

    def _create_reviewer_agent(self, role_name: str, persona: str, criteria_text: str, n_criteria: int) -> ChatAgent:
        """Create a reviewer agent with the given persona using BaseMessage."""
        task = _abstract_reviewer_task_prompt(
            self.population, self.exposure, self.comparison, self.outcome, criteria_text, n_criteria
        )

        sys_msg = BaseMessage.make_assistant_message(
            role_name=role_name,
            content=_abstract_reviewer_system_prompt(role_name, persona, task),
        )

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-5.4-nano",
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={"temperature": 0.2, "max_tokens": 10000},
        )

        return ChatAgent(system_message=sys_msg, model=model, token_limit=100000)

    async def _screen_paper_as_reviewer(
        self,
        reviewer_idx: int,
        agent: ChatAgent,
        paper_id: str,
        paper_content: str,
        n_criteria: int,
    ) -> Dict[str, Any]:
        """Screen one paper with one reviewer agent."""
        agent.reset()  # clear history from previous papers
        user_msg = f"Paper ID: {paper_id}\n\nContent: {paper_content}\n\nEvaluate this paper."
        try:
            response = await asyncio.to_thread(agent.step, user_msg)
            raw = response.msgs[0].content.strip()

            if raw.startswith("```json"):
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif raw.startswith("```"):
                raw = raw.split("```")[1].split("```")[0].strip()

            j_start = raw.find("{")
            j_end = raw.rfind("}")
            if j_start != -1 and j_end != -1:
                raw = raw[j_start:j_end + 1]

            parsed = json.loads(raw)
            evaluations = self._validate_evaluations(parsed.get("evaluations", []), n_criteria)

            num_yes = sum(1 for e in evaluations if e == "YES")
            decision = "INCLUDE" if (num_yes / n_criteria) >= self.include_threshold else "EXCLUDE"

            return {
                "reviewer_idx": reviewer_idx,
                "decision": decision,
                "evaluations": evaluations,
                "rationale": parsed.get("rationale", ""),
            }
        except Exception as e:
            return {
                "reviewer_idx": reviewer_idx,
                "decision": "ERROR",
                "evaluations": ["YES"] * n_criteria,
                "rationale": str(e),
            }

    async def _screen_paper_dual(
        self,
        agents: List[ChatAgent],
        paper_id: str,
        paper_content: str,
        n_criteria: int,
    ) -> Dict[str, Any]:
        """
        Screen one paper with both reviewer agents concurrently.
        Final decision: INCLUDE if at least one reviewer says INCLUDE.
        """
        r0, r1 = await asyncio.gather(
            self._screen_paper_as_reviewer(0, agents[0], paper_id, paper_content, n_criteria),
            self._screen_paper_as_reviewer(1, agents[1], paper_id, paper_content, n_criteria),
        )

        decisions = {r0["decision"], r1["decision"]}
        if "INCLUDE" in decisions:
            final = "INCLUDE"
        elif "ERROR" in decisions:
            final = "INCLUDE"
        else:
            final = "EXCLUDE"

        return {
            "paper_id": paper_id,
            "decision": final,
            "reviewer_1": r0,
            "reviewer_2": r1,
            "agreement": r0["decision"] == r1["decision"],
        }

    async def screen_papers_dual(
        self,
        input_csv: str,
        research_question: str,
        output_csv: str = None,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Screen papers with two independent reviewer agents.

        Personas are derived from the research question's most relevant domains.
        Decision rule: INCLUDE if at least one reviewer says INCLUDE.

        Args:
            input_csv: Path to CSV with 'id', 'title', 'abstract' columns.
            research_question: Used to derive the two reviewer personas.
            output_csv: Output path (default: input with '_dual_screened' suffix).
            batch_size: Max concurrent paper screenings.

        Returns:
            DataFrame with original columns plus dual-reviewer screening results.
        """
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        df = pd.read_csv(input_csv)

        if not self.content_criteria:
            self.generate_criteria()

        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(self.content_criteria))
        n_criteria = len(self.content_criteria)

        # Derive reviewer personas from research question
        print("\nDeriving reviewer personas from research question...")
        personas = self._derive_personas(research_question)
        agents = [
            self._create_reviewer_agent(p["role_name"], p["persona"], criteria_text, n_criteria)
            for p in personas
        ]

        print(f"\n{'=' * 60}")
        print(f"DUAL-REVIEWER SCREENING: {len(df)} papers")
        print(f"Reviewer 1: {personas[0]['role_name']}")
        print(f"Reviewer 2: {personas[1]['role_name']}")
        print(f"Decision rule: INCLUDE if at least one reviewer votes INCLUDE")
        print(f"Using {n_criteria} shared criteria")
        print(f"{'=' * 60}")

        papers = (df["title"] + ": " + df["abstract"].fillna("")).tolist()
        paper_ids = df["id"].astype(str).tolist()

        semaphore = asyncio.Semaphore(batch_size)

        async def limited_screen(i: int):
            async with semaphore:
                return await self._screen_paper_dual(
                    agents=agents,
                    paper_id=paper_ids[i],
                    paper_content=papers[i],
                    n_criteria=n_criteria,
                )

        all_results = await asyncio.gather(*[limited_screen(i) for i in range(len(papers))])

        n_include  = sum(1 for r in all_results if r["decision"] == "INCLUDE")
        n_exclude  = sum(1 for r in all_results if r["decision"] == "EXCLUDE")
        n_agree    = sum(1 for r in all_results if r["agreement"])

        print(f"\n{'=' * 60}")
        print(f"SCREENING COMPLETE")
        print(f"  Included : {n_include}")
        print(f"  Excluded : {n_exclude}")
        print(f"  Agreement: {n_agree}/{len(all_results)} ({100*n_agree//len(all_results)}%)")
        print(f"{'=' * 60}")

        result_map = {r["paper_id"]: r for r in all_results}
        df["decision"]           = df["id"].astype(str).map(lambda pid: result_map.get(pid, {}).get("decision", ""))
        df["reviewer_1_decision"]  = df["id"].astype(str).map(lambda pid: result_map.get(pid, {}).get("reviewer_1", {}).get("decision", ""))
        df["reviewer_1_rationale"] = df["id"].astype(str).map(lambda pid: result_map.get(pid, {}).get("reviewer_1", {}).get("rationale", ""))
        df["reviewer_2_decision"]  = df["id"].astype(str).map(lambda pid: result_map.get(pid, {}).get("reviewer_2", {}).get("decision", ""))
        df["reviewer_2_rationale"] = df["id"].astype(str).map(lambda pid: result_map.get(pid, {}).get("reviewer_2", {}).get("rationale", ""))
        df["agreement"]            = df["id"].astype(str).map(lambda pid: result_map.get(pid, {}).get("agreement", ""))

        if output_csv is None:
            output_csv = input_csv.replace("pubmed_results", "abstract_screen")

        df.to_csv(output_csv, index=False)
        print(f"\nSaved results to: {output_csv}")

        json_output = output_csv.replace(".csv", "_details.json")
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump({
                "peco": {"population": self.population, "exposure": self.exposure,
                         "comparison": self.comparison, "outcome": self.outcome},
                "criteria": self.content_criteria,
                "reviewers": [{"role_name": p["role_name"], "persona": p["persona"]} for p in personas],
                "decision_rule": "INCLUDE if at least one reviewer votes INCLUDE",
                "screening_results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed results to: {json_output}")

        included_df = df[df["decision"] == "INCLUDE"].copy()
        included_output = output_csv.replace(".csv", "_included.csv")
        included_df.to_csv(included_output, index=False)
        print(f"Saved {len(included_df)} included papers to: {included_output}\n")

        return df




async def run_screening(
    peco_elements_json: str,
    input_csv: str,
    research_question: str,
    output_csv: str = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Run dual-reviewer literature screening with PECO elements from JSON string.

    Args:
        peco_elements_json: JSON string containing PECO elements
        input_csv: Path to input CSV with papers
        research_question: Research question used to derive reviewer personas
        output_csv: Path to save screening results (optional)
        batch_size: Max concurrent paper screenings (default: 32)

    Returns:
        DataFrame with screening results
    """
    peco_elements = parse_peco_json(peco_elements_json)
    peco_values = extract_peco_values(peco_elements)

    print(f"\n{'=' * 60}")
    print("PECO Elements for Literature Screening:")
    print(f"{'=' * 60}")
    print(f"Population: {peco_values['population']}")
    print(f"Exposure: {peco_values['exposure']}")
    print(f"Comparison: {peco_values['comparison']}")
    print(f"Outcome: {peco_values['outcome']}")
    print(f"Mode: Dual-reviewer (max {batch_size} concurrent)")
    print(f"{'=' * 60}\n")

    screener = DualReviewerScreener(
        exposure=peco_values['exposure'],
        outcome=peco_values['outcome'],
        population=peco_values['population'],
        comparison=peco_values['comparison'],
    )

    return await screener.screen_papers_dual(
        input_csv=input_csv,
        research_question=research_question,
        output_csv=output_csv,
        batch_size=batch_size,
    )


async def main(
    peco_elements_json: str,
    input_csv: str,
    research_question: str,
    output_csv: str = None,
    batch_size: int = 32,
):
    """
    Main function for command-line usage

    Args:
        peco_elements_json: JSON string containing PECO elements from previous agent step
        input_csv: Path to input CSV with papers
        research_question: Research question used to derive reviewer personas
        output_csv: Path to save screening results
        batch_size: Max concurrent paper screenings (default: 32)
    """
    return await run_screening(
        peco_elements_json=peco_elements_json,
        input_csv=input_csv,
        research_question=research_question,
        output_csv=output_csv,
        batch_size=batch_size,
    )


# Default PECO JSON for testing


DEFAULT_PECO_JSON = ""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Literature Screening with PECO elements")
    parser.add_argument(
        "--peco-json",
        type=str,
        default=None,
        help="JSON string containing PECO elements (population, exposure, comparison, outcome)"
    )
    parser.add_argument(
        "--peco-file",
        type=str,
        default=None,
        help="Path to JSON file containing PECO elements"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Path to input CSV with papers (must have 'id', 'title', 'abstract' columns)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Path to save screening results (defaults to input_csv with '_screened' suffix)"
    )
    parser.add_argument(
        "--research-question",
        type=str,
        default="",
        help="Research question used to derive reviewer personas"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Max concurrent paper screenings (default: 32)"
    )

    args = parser.parse_args()

    # Load PECO JSON from file if provided, otherwise use command line arg or default
    peco_json = args.peco_json
    if args.peco_file and os.path.exists(args.peco_file):
        with open(args.peco_file, 'r', encoding='utf-8') as f:
            peco_json = f.read()
    elif peco_json is None:
        peco_json = DEFAULT_PECO_JSON

    asyncio.run(main(
        peco_elements_json=peco_json,
        input_csv=args.input_csv,
        research_question=args.research_question,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
    ))
