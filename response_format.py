from pydantic import BaseModel, Field
from typing import List, Optional, Union

class PECOOperatorGroup(BaseModel):
            within_group: str
            between_groups: str

class PECOItem(BaseModel):
    raw_term: str
    terms: List[str]
    standard_mesh_term: Optional[List[str]] = Field(description="MeSH terms found via look_up_tool, or null if lookup failed")
    scope_note: Optional[str] = None
    group_id: int

class PECOOperators(BaseModel):
    population: Union[str, PECOOperatorGroup]
    exposure: Union[str, PECOOperatorGroup]
    comparison: Union[str, PECOOperatorGroup]
    outcome: str

class PECOExtraction(BaseModel):
    population: List[PECOItem]
    exposure: List[PECOItem]
    comparison: List[PECOItem]
    outcome: List[PECOItem]
    peco_operators: PECOOperators