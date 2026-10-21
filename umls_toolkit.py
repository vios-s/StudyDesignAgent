import os
import json
from dotenv import load_dotenv
from umls_python_client import UMLSClient
load_dotenv()


class UMLSToolkit:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("UMLS_API_KEY")
        self.umls_client = UMLSClient(api_key=self.api_key)

    def _parse_response(self, response, context=""):
        """
        Parse API response - handles string, dict, and None types.

        Args:
            response: API response (could be str, dict, or None)
            context: Context string for error messages

        Returns:
            dict or None
        """
        if response is None:
            print(f"Warning: API returned None for {context}")
            return None

        if isinstance(response, dict):
            return response

        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                print(f"Warning: API returned non-JSON string for {context}: {response[:200]}")
                return None

        print(f"Warning: Unexpected response type {type(response)} for {context}")
        return None

    def search_terms(self, query, sabs=None, page_size=1):
        """Search for terms in UMLS"""
        ui_list = []
        search_api = self.umls_client.searchAPI

        try:
            response = search_api.search(query, sabs=sabs, page_size=page_size)
            results = self._parse_response(response, f"search query '{query}'")

            if results:
                # Navigate nested structure: results["result"]["results"]
                result_dict = results.get("result", {})
                result_list = result_dict.get("results", [])

                if result_list:
                    for result in result_list:
                        if isinstance(result, dict):
                            ui = result.get("ui", "")
                            if ui:
                                ui_list.append(ui)
                else:
                    print(f"Info: No results found for query: {query}")

        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")

        return ui_list
    
    def get_cui_info(self, cui, sabs=None, language="ENG", atoms_num=25, definitions_num=3):
        atom_result = []
        defintion_result = []
        cui_api = self.umls_client.cuiAPI

        try:
            # Get CUI info - navigate nested structure
            cui_response = self._parse_response(
                cui_api.get_cui_info(cui),
                f"CUI info for '{cui}'"
            )
            cui_info = cui_response.get("result", {}) if cui_response else {}
            # Extract only the 'name' from semanticTypes
            semantic_types_raw = cui_info.get("semanticTypes", [])
            sementic_types = [st.get("name", "") for st in semantic_types_raw if isinstance(st, dict)]

            # Get definitions - navigate nested structure
            defs_response = self._parse_response(
                cui_api.get_definitions(cui, sabs=sabs, page_size=definitions_num),
                f"definitions for CUI '{cui}'"
            )
            if defs_response:
                defs_list = defs_response.get("result", [])
                for result in defs_list:
                    if isinstance(result, dict):
                        defintion_result.append(result.get("value", ""))

        except Exception as e:
            print(f"Error getting CUI info for '{cui}': {str(e)}")

        return {
            "cui": cui,
            "semantic_types": sementic_types,
            "definitions": defintion_result
        }
    
    def get_atoms(self, cui, sabs=None, language="ENG", page_size=25):
        """Get atoms for a given CUI"""
        atom_result = []
        cui_api = self.umls_client.cuiAPI

        try:
            # Get atoms - navigate nested structure
            atoms_response = self._parse_response(
                cui_api.get_atoms(cui, page_size=page_size, language=language, sabs=sabs),
                f"atoms for CUI '{cui}'"
            )
            if atoms_response:
                atoms_list = atoms_response.get("result", {})
                for result in atoms_list:
                    if isinstance(result, dict):
                        atom_result.append((
                            result.get("name", ""),
                            result.get("rootSource", ""),
                            result.get("code", "").split("/")[-1]
                        ))

        except Exception as e:
            print(f"Error getting atoms for '{cui}': {str(e)}")

        return atom_result
     
    def get_parent_child_pathways(self, source_id, source="SNOMEDCT_US", max_depth=0):
        """Get parent and child pathways for a given CUI"""
        source_api = self.umls_client.sourceAPI
        parents = []
        children = []

        try:
            # Call with correct parameter order: source first, then source_id
            pathway_response = self._parse_response(
                source_api.get_concept_pathways(source, source_id, max_depth=max_depth),
                f"pathways for '{source_id}'"
            )

            if pathway_response:
                # Navigate nested structure if present (fallback to response itself)
                # pathway = pathway_response.get("result", pathway_response)
                parents = pathway_response.get(f"concept_{source_id}_parents", [])
                children = pathway_response.get(f"concept_{source_id}_children", [])

        except Exception as e:
            print(f"Error getting pathways for '{source_id}': {str(e)}")

        return {
            "parents": parents,
            "children": children
        }

