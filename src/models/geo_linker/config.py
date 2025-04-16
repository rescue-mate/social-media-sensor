from typing import Optional

from pydantic_settings import BaseSettings


class GeoLinkerSettings(BaseSettings):
    num_examples_per_class: int = 8
    batch_size: int = 24
    photon_url: str = "http://134.100.39.20:2322/api/"
    nominatim_url: str = "http://134.100.39.20:8080/search"
    default_city: Optional[str] = None
    default_country: Optional[str] = "Deutschland"
    default_district: Optional[str] = None
    num_candidates: int = 3
    do_collective_linking: bool = False
    do_llm_filtering: bool = False