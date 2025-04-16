from abc import abstractmethod, ABC
from typing import List, Dict

from sklearn.preprocessing import MultiLabelBinarizer


class TextClassificationModel(ABC):
    def __init__(self):
        self.target_names = ['affected_individual', 'caution_and_advice',
            'displaced_and_evacuations', 'donation_and_volunteering',
            'infrastructure_and_utilities_damage', 'injured_or_dead_people',
            'missing_and_found_people', 'not_humanitarian',
            'requests_or_needs', 'response_efforts', 'sympathy_and_support']
        self.mlb = MultiLabelBinarizer(classes=self.target_names)
        self.mlb.fit([self.target_names])
        self.num_labels = len(self.target_names)


    @abstractmethod
    def predict(self, texts: List[str]) -> List[Dict]:
        pass