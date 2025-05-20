import enum
import logging
from itertools import product
from typing import List, Dict, Set
from geopy.distance import geodesic, great_circle
import requests
from nervaluate import Evaluator
import numpy as np
import re
from fuzzywuzzy import fuzz
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import faiss
import dataclasses
import random
import json
import argparse

from src.models.geo_linker.config import FewShots
from src.models.utils import LLMWrapper
import requests_cache
from haversine import haversine_vector, Unit
logger =  logging.getLogger(__name__)

@dataclasses.dataclass
class IndexTuple:
    index: faiss.Index
    mapping: Dict[int, int]
class GeoLinker:
    def __init__(self,
                 model: LLMWrapper,
                 few_shot_data: List = None,
                    num_examples_per_class: int = 8,
                    balanced_few_shots: bool = True,
                    batch_size: int = 24,
                    photon_url: str = "https://photon.komoot.io/api/",
                    nominatim_url: str = "https://nominatim.openstreetmap.org/search",
                    default_city: str = None,
                    default_country: str = None,
                    default_district: str = None,
                    do_llm_filtering: bool = False,
                    do_collective_linking: bool = False,
                    do_llm_based_disambiguation: bool = True,
                    few_shots: FewShots = FewShots.STATIC,
                    num_candidates: int = 3
                        ):

        if few_shot_data is None:
            test, dev, train = extract_data("./data/geo_linking_dataset/")
            few_shot_data = train + dev + test

        self.labels = self.get_entity_labels(few_shot_data)
        self.balanced_few_shots = balanced_few_shots
        self.in_context_examples, self.in_context_indexes, self.in_context_model,  = self.load_in_context_examples(few_shot_data)
        self.main_prompt = open("data/geo_link_prompt.txt").read()
        self.main_fixed_few_shots = open("data/geo_link_examples.txt").read()
        self.disambiguation_prompt = open("data/geo_disambiguation_prompt.txt").read()
        self.num_examples_per_class = num_examples_per_class
        self.batch_size = batch_size
        self.few_shots = few_shots
        self.model = model
        self.nominatim_url = nominatim_url
        self.photon_url = photon_url
        self.default_city = default_city
        self.default_country = default_country
        self.default_district = default_district
        self.do_llm_filtering = do_llm_filtering
        self.do_collective_linking = do_collective_linking
        assert not do_collective_linking, "Collective linking is currently not supported"
        self.num_candidates = num_candidates
        self.do_llm_based_disambiguation = do_llm_based_disambiguation


    @staticmethod
    def get_entity_labels(data: List[Dict]) -> Set[str]:
            '''
            Extracts all unique entity type labels from the annotations in the data.
            '''
            entity_labels = set()
            for item in data:
                for annotation in item.get("annotations", []):
                    entity_labels.add(annotation[1])  

            return entity_labels

    @staticmethod
    def convert_annotations(data: List[Dict], test=False) -> List[Dict]:
        '''
        Converts annotations from [['George Washington', 'PER'], ['Washington', 'LOC']] in thr jsonl file
        to pridogy style for evaluation: [{"label": "PER", "start": 2, "end": 4}, {"label": "LOC", "start": 1, "end": 2}]
        '''
        converted = []
        for sample in data:
            sent = sample["text"]
            entities_per_sentence = []

            if test:
                entities = sample["pred"]
            else:
                entities = sample["annotations"]

            for entity in entities:
                entities_per_sentence.append({
                    "label": entity[1],
                    "start": sent.find(entity[0]),
                    "end": sent.find(entity[0]) + len(entity[0])
                })
            converted.append(entities_per_sentence)
        assert len(converted) == len(data)
        return converted

    @staticmethod
    def load_in_context_examples(data: List[Dict]):
        all_examples = {}
        sentences = []
        per_class = defaultdict(list)
        counter = 0
        for elem in data:
            class_labels = [anno[1] for anno in elem["annotations"]]
            for class_label in class_labels:
                all_examples[counter] = {
                                        "text": elem["text"],
                                        "class": class_label,
                                        "annotation": elem["annotations"]
                                        }
                sentences.append(elem["text"])
                per_class[class_label].append(counter)
                counter += 1
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(sentences, show_progress_bar=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        indexes = {"all": IndexTuple(index, {idx: idx for idx in range(len(sentences))})}

        for class_label, indices in per_class.items():
            indices = np.array(indices)
            embeddings_class = embeddings[indices]
            index_class = faiss.IndexFlatL2(embeddings_class.shape[1])
            index_class.add(embeddings_class)
            indexes[class_label] = IndexTuple(index_class, {idx: indices[idx] for idx in range(len(indices))})

            return all_examples, indexes, model

    def get_in_context_examples(self, texts: List[str], scope: str = "all", num: int = 5):
        embeddings = self.in_context_model.encode(texts, show_progress_bar=False)

        D, I = self.in_context_indexes[scope].index.search(embeddings, num)

        contexts = []
        for idx in range(len(texts)):
            contexts.append([self.in_context_examples[self.in_context_indexes[scope].mapping[i]] for i in I[idx]])
        return contexts

    def form_prompt(self, data: List[Dict]) -> List[list]:
        texts = [sample["text"] for sample in data]

        if not texts:
            return []
        in_context_examples = self.get_in_context_examples(texts, scope="all", num=self.num_examples_per_class)


        test_prompts = []
        for sample, context in tqdm(zip(data, in_context_examples)):
            complete_prompt = self.main_prompt
            if self.few_shots == FewShots.STATIC or self.few_shots == FewShots.BOTH:
                complete_prompt += self.main_fixed_few_shots
            if self.few_shots == FewShots.DYNAMIC or self.few_shots == FewShots.BOTH:
                complete_prompt += create_examples(context)
            test_prompt = [{"role": "system", "content": complete_prompt},
                      {"role": "user", "content": "Input: " + sample["text"]}]

            test_prompts.append(test_prompt)
        return test_prompts

    def identify_locations(self, data: List[Dict]):
        '''
        Runs the LLM model on the test
        '''
        
        messages = self.form_prompt(data)   
        result = self.model.generate(messages)
        return self._parse_llm_outputs_(result)
    
    
    @staticmethod
    def _parse_llm_outputs_(outputs)-> List[List[dict]]:
        results = []
        for generated_text in outputs:
            skimmed_generated_text = generated_text[generated_text.rfind("Extracted Entities") + len("Extracted Entities"):]
            bracket_left = skimmed_generated_text.find("[")
            bracket_right = skimmed_generated_text.rfind("]")
            if bracket_left != -1 and bracket_right != -1:
                skimmed_generated_text = skimmed_generated_text[bracket_left: bracket_right + 1]

                try:
                    content = json.loads(skimmed_generated_text)
                except:
                    try:
                        content = eval(skimmed_generated_text)
                    except:
                        logger.warning(f"Error parsing JSON in entity recognition: {generated_text}")
                        content = []
                if not isinstance(content, list):
                    logger.warning(f"Error parsing JSON in entity recognition: {generated_text}")
                    content = []
            else:
                logger.warning(f"Error parsing JSON in entity recognition: {generated_text}")
                content = []
            filtered_results = []
            for result in content:
                if not isinstance(result, dict):
                    continue
                if "entity" not in result or "type" not in result:
                    continue
                if not isinstance(result["entity"], str) or not isinstance(result["type"], str):
                    continue
                filtered_results.append(result)
            results.append(filtered_results)

        return results

    def evaluate(self, true_labels:list, pred:list):
        '''
        Evaluates the predicted results using precision, recall, and F1-score.
        '''
        assert len(true_labels) == len(pred)

        evaluator = Evaluator(true_labels, pred, tags= list(self.labels))
        results, results_per_tag, _, _ = evaluator.evaluate()

        print(results)
        print(results_per_tag)
        return results, results_per_tag

    def get_nominatim_locations(self, osm_ids: list):
        #   https://nominatim.openstreetmap.org/lookup?osm_ids=[N|W|R]<value>,…,…,&<params>

        url = "https://nominatim.openstreetmap.org/lookup"

        params = {
            "osm_ids": ",".join(osm_ids),
            "format": "json"
        }

        headers = {
            "User-Agent": "RM"  # Replace with your app info
        }

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            elements = []
            already_seen = set()
            for element in response.json():
                if element["osm_id"] in already_seen:
                    continue
                already_seen.add(element["osm_id"])
                elements.append(element)
            return elements

        else:
            raise Exception(f"Error {response.status_code}: {response.text}")


    def find_nominatim_candidates(self, address: str, district: str = None, city: str = "Hamburg", country: str = "Germany"):
        url = self.nominatim_url
        params = {
            "q": address,
            "format": "json",
            "limit": 39,
        }

        headers = {
            "User-Agent": "RM"  # Replace with your app info
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data:
                return data  # Return the first result
            else:
                return []


    def find_photon_candidates(self, address: str, district: str, city: str, country: str):
        params = {
            "q": address,
            "limit": 50,
            "osm_tag": "!historic",
        }

        headers = {
            "User-Agent": "RM"  # Replace with your app info
        }

        response = requests.get(self.photon_url, params=params, headers=headers)
        found_candidates = []
        if response.status_code == 200:
            data = response.json()
            for feature in data["features"]:
                if district:
                    if feature["properties"].get("district") != district:
                        continue
                if city:
                    if feature["properties"].get("city") != city:
                        continue
                if country:
                    if feature["properties"].get("country") != country:
                        continue

                found_candidates.append(feature)

        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

        return found_candidates

    @staticmethod
    def clean(location_param: str, replacement):
        if isinstance(location_param, str):
            location_param = location_param.replace("...", "")
            if not(location_param.strip()):
                return replacement
            return location_param.strip()
        return replacement

    def get_geo_candidates(self, location_mentions: List[dict], district: str, city: str, country: str, batch_size=20):
        all_nominatim_candidates = []
        all_photon_candidates = []
        for location in location_mentions:
            address = location["entity"]
            selected_district = self.clean(location.get("district", district), district)
            selected_city = self.clean(location.get("city", city), city)
            selected_street = self.clean(location.get("street"), None)
            # selected_country = self.clean(location.get("country", country), country)
            if selected_street and selected_street != location["entity"]:
                address += f" {selected_street}"
            if selected_district and selected_district != location["entity"]:
                address += f" {selected_district}"
            if selected_city and selected_city != location["entity"]:
                address += f" {selected_city}"
            # if selected_country and selected_country != location["entity"]:
            #     address += f", {selected_country}"

            try:
                photon_candidates = self.find_photon_candidates(address, district, city, country)
            except Exception as e:
                print(f"Error for {address}: {e}")
                photon_candidates = []

            try:
                nominatim_candidates = self.find_nominatim_candidates(address, district, city, country)
            except Exception as e:
                print(f"Error for {address}: {e}")
                nominatim_candidates = []
            osm_ids_in_nominatim = set()
            for feature in nominatim_candidates:
                osm_ids_in_nominatim.add(f"{feature['osm_type']}{feature['osm_id']}")
            all_nominatim_candidates.append(nominatim_candidates)
            photon_candidates = [x for x in photon_candidates if f"{x['properties']['osm_type']}{x['properties']['osm_id']}" not in osm_ids_in_nominatim]
            all_photon_candidates.append(photon_candidates)



        final_candidates = []
        for nominatim_candidates, photon_candidates in zip(all_nominatim_candidates, all_photon_candidates):
            osm_ids = []
            for feature in photon_candidates:
                osm_ids.append(f"{feature['properties']['osm_type']}{feature['properties']['osm_id']}")
            retrieved_nominatim_candidates = []
            for i in range(0, len(osm_ids), batch_size):
                nominatim_candidates += self.get_nominatim_locations(osm_ids[i:i + batch_size])

            final_candidates.append(nominatim_candidates + retrieved_nominatim_candidates)

        return final_candidates

    def do_bboxes_overlap(self, location1, location2):
        """Checks if two bounding boxes overlap."""
        min_lat1, max_lat1, min_lon1, max_lon1 = location1
        min_lat2, max_lat2, min_lon2, max_lon2 = location2

        # Check for overlap in latitude and longitude ranges
        return not (max_lat1 < min_lat2 or min_lat1 > max_lat2 or max_lon1 < min_lon2 or min_lon1 > max_lon2)


    def calculate_distance_between_locations(self, location1, location2):
        if self.do_bboxes_overlap(location1, location2):
            return 0.0
        location1 = [
            (location1[0], location1[2]), (location1[0], location1[3]),
            (location1[1], location1[2]), (location1[1], location1[3])
        ]
        location2 = [
            (location2[0], location2[2]), (location2[0], location2[3]),
            (location2[1], location2[2]), (location2[1], location2[3])
        ]
        distances = [great_circle(p1, p2).meters for p1, p2 in product(location1, location2)]
        return min(distances) / 1000

    def find_best_assignments(self, candidate_locations, candidate_relevances):
        distances = {}

        max_num_locations = max(candidate_locations.keys()) + 1
        candidate_indices = []
        for i in range(max_num_locations):
            if i not in candidate_locations:
                continue
            candidate_locations_a = candidate_locations[i]
            candidate_indices_for_i = []
            for k in range(len(candidate_locations[i])):
                candidate_indices_for_i.append((i, k))
            candidate_indices.append(candidate_indices_for_i)
            candidate_locations_a = np.array(candidate_locations_a)
            for j in range(i + 1, max_num_locations):
                if j not in candidate_locations:
                    continue
                candidate_locations_b = np.array(candidate_locations[j])

                candidate_locations_a_pair = candidate_locations_a[:, None, :].repeat(candidate_locations_b.shape[0], axis=1).reshape(-1, candidate_locations_a.shape[-1])
                candidate_locations_b_pair = candidate_locations_b[None, :, :].repeat(candidate_locations_a.shape[0], axis=0).reshape(-1, candidate_locations_a.shape[-1])

                dist_1 = haversine_vector(np.stack((candidate_locations_a_pair[:, 0], candidate_locations_a_pair[:, 2]), axis=-1),
                                   np.stack((candidate_locations_b_pair[:, 0], candidate_locations_b_pair[:, 2]), axis=-1), unit=Unit.KILOMETERS)

                dist_2 = haversine_vector(np.stack((candidate_locations_a_pair[:, 0], candidate_locations_a_pair[:, 3]), axis=-1),
                                      np.stack((candidate_locations_b_pair[:, 0], candidate_locations_b_pair[:, 3]), axis=-1), unit=Unit.KILOMETERS)

                dist_3 = haversine_vector(np.stack((candidate_locations_a_pair[:, 1], candidate_locations_a_pair[:, 2]), axis=-1),
                                   np.stack((candidate_locations_b_pair[:, 1], candidate_locations_b_pair[:, 2]), axis=-1),
                                   unit=Unit.KILOMETERS)

                dist_4 = haversine_vector(np.stack((candidate_locations_a_pair[:, 1], candidate_locations_a_pair[:, 3]), axis=-1),
                                   np.stack((candidate_locations_b_pair[:, 1], candidate_locations_b_pair[:, 3]), axis=-1),
                                   unit=Unit.KILOMETERS)

                dist = np.stack([dist_1, dist_2, dist_3, dist_4], axis=-1).min(axis=-1)

                overlapping = ~np.logical_or(np.logical_or(candidate_locations_a_pair[:, 1] <= candidate_locations_b_pair[:, 0]
                                , candidate_locations_a_pair[:, 0] >= candidate_locations_b_pair[:, 1]),
                                np.logical_or(candidate_locations_a_pair[:, 3] <= candidate_locations_b_pair[:, 2]
                                ,candidate_locations_a_pair[:, 2] >= candidate_locations_b_pair[:, 3]))
                dist[overlapping] = 0



                distances[(i, j)] = dist.reshape(candidate_locations_a.shape[0], candidate_locations_b.shape[0])
        assignments = []
        for prod in product(*candidate_indices):
            full_distance = 0
            full_relevance = 0
            for idx, elem_1 in enumerate(prod[:-1]):
                for elem_2 in prod[idx + 1:]:
                    i, k = elem_1
                    j, l = elem_2
                    if i > j:
                        i, j = j, i
                        k, l = l, k
                    distance = distances.get((i, j))[k, l]
                    full_distance += distance
                    full_relevance += candidate_relevances[i][k] + candidate_relevances[j][l]
            assignments.append(((full_distance, -full_relevance), prod))

        assignments = sorted(assignments, key=lambda x: x[0])

        return assignments
    def llm_based_disambiguation(self, texts, all_candidates, predictions):
        messages = []
        text_to_response = defaultdict(list)
        for text_idx, (text, candidate_lists, predictions_list) in enumerate(zip(texts, all_candidates, predictions)):
            for mention_idx, candidates in enumerate(candidate_lists):
                user_input = "Input: " + text
                if not candidates:
                    text_to_response[text_idx].append((None, mention_idx))
                    continue
                prediction = predictions_list[mention_idx]
                user_input += f"\nLocation: {prediction['entity']}"
                user_input += f"\nCandidate Locations:\n"
                for idx, location in enumerate(candidates[:20]):
                    user_input += f"{idx + 1}: {location['display_name']}\n"

                message = [
                    {"role": "system", "content": self.disambiguation_prompt},
                           {"role": "user", "content": user_input}]
                text_to_response[text_idx].append((len(messages), mention_idx))
                messages.append(message)
        responses = self.model.generate(messages)

        final_locations = []
        for idx in range(len(texts)):
            if idx not in text_to_response:
                final_locations.append([])
                continue
            candidate_lists = all_candidates[idx]
            locations = []
            for response_idx, mention_idx in text_to_response[idx]:
                candidates = candidate_lists[mention_idx]
                output_location = {
                    "mention": predictions[idx][mention_idx]["entity"],
                }
                if response_idx is None:
                    output_location["location"] = None
                else:
                    response = responses[response_idx]
                    bracket_start = response.find("{")
                    bracket_end = response.rfind("}")
                    correct_candidate = None
                    if bracket_start != -1 and bracket_end != -1:
                        try:
                            content = json.loads(response[bracket_start:bracket_end + 1])
                        except json.JSONDecodeError:
                            try:
                                content = eval(response[bracket_start:bracket_end + 1])
                            except:
                                logger.warning(f"Error parsing JSON in entity disambiguation: {response}")
                                content = None
                        if content and "Correct candidate" in content:
                            if content["Correct candidate"] is None:
                                correct_candidate = None
                            else:
                                correct_candidate = content["Correct candidate"]
                                try:
                                    correct_candidate = int(correct_candidate) - 1
                                    if correct_candidate < 0:
                                        correct_candidate = None
                                    if correct_candidate >= len(candidates):
                                        correct_candidate = None
                                except ValueError:
                                    correct_candidate = None
                                    logger.warning(f"Error parsing JSON in entity disambiguation: {response}")
                    else:
                        logger.warning(f"Error parsing JSON in entity disambiguation: {response}")
                    if correct_candidate is None:
                        output_location["location"] = None
                    else:
                        output_location["location"] = candidates[correct_candidate]
                locations.append(output_location)
            final_locations.append(locations)
        return final_locations



    def llm_based_filtering(self, texts, locations):
        messages = []
        text_to_response = {}
        for text_idx, (text, locations_) in enumerate(zip(texts, locations)):
            user_input = "Input: " + text
            if not locations_:
                continue
            user_input += f"\nExtracted Locations:\n"
            for idx, location in enumerate(locations_):
                user_input += f"{idx + 1}: {location['name']}, {location['display_name']}\n"

            message = [{"role": "system",
                        "content": "Given is an input text and locations. Decide for each location whether it is actually mentioned in the text. "
                                   "\nEach location is represented by a counter, its name and the full address. The input looks as follows:"
                                   "\n\nInput: ..."
                                   "\nExtracted Locations: "
                                   "\n1: <name_1>, <display_name_1>"
                                   "\n2: <name_2>, <display_name_2>"
                                   "\n..."
                                   "\n\nPlease return everything in the following format: "
                                   "\n\nExplanation: Location 1 is mentioned in the text because ..., Location 3 is mentioned in the text because ..., Location 4 is not mentioned in the text because ..."
                                   "\nExtracted Entities: [1, 3, ...]"},
                       {"role": "user", "content": user_input}]
            text_to_response[text_idx] = len(messages)
            messages.append(message)
        if not messages:
            return locations
        responses = self.model.generate(messages)

        final_locations = []
        for locations_idx, locations_ in enumerate(locations):
            if locations_idx not in text_to_response:
                final_locations.append(locations_)
                continue
            response = responses[text_to_response[locations_idx]]
            response = response[response.rfind("Extracted Entities:") + len("Extracted Entities:"):].strip()
            try:
                filtered_locations = eval(response)
                filtered_locations = [locations_[idx - 1] for idx in filtered_locations if idx - 1 < len(locations_)]
            except Exception as e:
                filtered_locations = locations_

            final_locations.append(filtered_locations)

        return final_locations
    def llm_based_selection(self, texts, locations):
        messages = []
        text_to_response = {}
        for text_idx, (text, locations_) in enumerate(zip(texts, locations)):
            user_input = "Input: " + text
            if not locations_:
                continue
            for idx, assignment in enumerate(locations_):
                user_input += f"\n\nSet {idx + 1}"
                user_input += f"\nExtracted Locations:\n"
                for location in enumerate(assignment):
                    user_input += f"{location[1]['name']} | {location[1]['addresstype']} | {location[1]['display_name']}\n"

            message =  [{"role": "system", "content": "Given is an input text and multiple possible extracted sets of locations. Decide which set is the most likely one based on the information given for each entity (especially the type). Give an explanation for your decision. "
                                                      "\nEach location is represented by its name, its type and the full address. The input looks as follows:"
                                                      "\n\nInput: ..."
                                                      "\n\nSet 1"
                                                      "\nExtracted Locations: "
                                                      "\n<name> | <type> | <display_name> "
                                                      "\n<name> | <type> | <display_name>"
                                                      "\n..."
                                                      "\n\nSet 2"
                                                      "\nExtracted Locations: "
                                                      "\n<name> | <type> | <display_name>"
                                                      "\n..."
                                                      "\n\nPlease return everything in the following format: "
                                                      "\n\nExplanation: It is set ... because ..."
                                                      "\n\nMost likely set: <set_number>"},
                        {"role": "user", "content": user_input}]
            text_to_response[text_idx] = len(messages)
            messages.append(message)
        if not messages:
            return locations
        responses = self.model.generate(messages)

        final_locations = []
        for locations_idx, locations_ in enumerate(locations):
            if locations_idx not in text_to_response:
                final_locations.append([])
                continue
            response = responses[text_to_response[locations_idx]]
            response = response[response.rfind("Most likely set:") + len("Most likely set:"):].strip()
            try:
                most_likely_assignment = eval(response)
                most_likely_assignment = most_likely_assignment - 1
                filtered_locations = locations_[most_likely_assignment]
            except Exception as e:
                filtered_locations = locations_[0]

            final_locations.append(filtered_locations)

        return final_locations

    def disambiguate_candidates(self, texts: List[str], all_candidates: List[List[Dict]], predictions: List[List[dict]]):

        final_locations = []
        if self.do_collective_linking:
            # TODO: Not yet adapted to new ouput
            for candidates, predictions_ in zip(all_candidates, predictions):
                candidate_locations = defaultdict(list)
                candidate_relevances = defaultdict(list)
                for idx, candidate_set in enumerate(candidates):
                    for candidate in candidate_set:
                        bounding_box = candidate["boundingbox"]
                        bounding_box = [float(coord) for coord in bounding_box]
                        candidate_locations[idx].append(bounding_box)
                        candidate_relevances[idx].append(candidate["importance"])
                if not candidate_locations:
                    final_locations.append([])
                    continue
                assignments = self.find_best_assignments(candidate_locations, candidate_relevances)[:self.num_candidates]
                new_assignments = []
                for assignment in assignments:
                    asssigned_locations = []
                    for k, v in sorted(assignment[1]):
                        if candidates[k]:
                            asssigned_locations.append(candidates[k][v])
                    new_assignments.append(asssigned_locations)
                final_locations.append(new_assignments)

            final_locations = self.llm_based_selection(texts, final_locations)
        elif self.do_llm_based_disambiguation:
            final_locations = self.llm_based_disambiguation(texts, all_candidates, predictions)
        else:
            # Sort by candidate_relevances
            for candidates, predictions_ in zip(all_candidates, predictions):
                naive_selection = []
                for candidate_set, prediction in zip(candidates, predictions_):
                    if candidate_set:
                        candidate_set = sorted(candidate_set, key=lambda x: x["importance"], reverse=True)
                        candidate = candidate_set[0]
                    else:
                        candidate = None
                    naive_selection.append({
                        "mention": prediction["entity"],
                        "location": candidate,
                    })
                final_locations.append(naive_selection)

        if self.do_llm_filtering:
            final_locations = self.llm_based_filtering(texts, final_locations)


        return final_locations


    def link_texts(self, texts: List[str], district: str = None, city: str = None, country: str = None):
        logger.info("Geoparsing texts...")
        if district is None:
            district = self.default_district
        if city is None:
            city = self.default_city
        if country is None:
            country = self.default_country

        predictions = self.identify_locations([{"text": text} for text in texts])

        logger.info("Getting geo candidates...")
        candidates_for_all = []
        for prediction in predictions:
            candidates = self.get_geo_candidates(prediction, district, city, country)
            candidates_for_all.append(candidates)

        logger.info("Disambiguating candidates...")
        locations_list = self.disambiguate_candidates(texts, candidates_for_all, predictions)

        logger.info("Finished geoparsing texts.")
        return locations_list

    def eval_main(self, test_dataset):
        '''
        run evaluation on the test data
        '''
        predictions = self.identify_locations(test_dataset)
        print(predictions[:10])
        true_labels = self.convert_annotations(test_dataset, test=False)
        pred = self.convert_annotations(predictions, test=True)

        self.evaluate(true_labels=true_labels, pred=pred)
        return 
        
    def inference(self, text: str):
        '''
        given any text, return the span
        '''
        sample = {"text": text}
        predictions = self.identify_locations([sample])
        return [entities[0] for entities in predictions[0]["pred"]]


def create_examples(context: list):
    '''
    helper function to create examples for the prompt
    '''
    random.shuffle(context)
    prompt = ""
    for idx, elem in enumerate(context):
        text = elem["text"]
        text=text.strip()
        prompt += f"Input: {text}\nExplanation:\n...\n\nExtracted Entities:\n"
        list_of_entities = []
        for ann_idx, anno in enumerate(elem["annotation"]):
            list_of_entities.append({
                "entity": anno[0],
                "type": anno[1],
                "street": "...",
                "district": "...",
                "city": "...",
                "country": "...",
            })
        prompt += json.dumps(list_of_entities, indent=4)
        prompt += "\n\n"
    return prompt

def extract_data(file_dir:str):
    '''
    return test, train and dev data
    '''
    test, dev, train = file_dir+"test_processed.jsonl", \
                           file_dir+"dev_processed.jsonl", \
                           file_dir+"train_processed.jsonl"
    test, dev, train = read_file(test), read_file(dev), read_file(train)
    return test, dev, train

def read_file(file_address:str)->List[Dict]:
        '''
        out: List of dictionaries, {"text": str, "annotation": list[list]}
        '''
        with open(file_address, "r") as file:
            return [json.loads(line) for line in file]
