import enum
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class FewShots(enum.Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    BOTH = "both"



class GeoLinkerSettings(BaseSettings):
    num_examples_per_class: int = 8
    batch_size: int = 24
    photon_url: str = "http://134.100.39.20:2322/api/"
    nominatim_url: str = "http://134.100.39.20:8080/search"
    default_city: Optional[str] = None
    default_country: Optional[str] = None
    default_district: Optional[str] = None
    num_candidates: int = 3
    do_collective_linking: bool = False
    do_llm_filtering: bool = False
    few_shots: FewShots = FewShots.STATIC

    @field_validator("default_city", "default_country", "default_district", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        return None if v == "" else v