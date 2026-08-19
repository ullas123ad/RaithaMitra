"""
Mis-Mapping & Collision Matrix Verification Suite
=================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru

Validates that easily confused, phonetic, semantic, and linguistic crop pairs
across Kannada and English resolve strictly to their correct canonical crop IDs
with 0% alias overlap, 0% mis-mapping, and 0% collision.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.advisory.crop_identifier import (
    normalize_crop_name,
    detect_crop_from_text,
    resolve_canonical_crop,
    get_crop_entry,
    get_crop_support_status,
    get_karnataka_suitability,
)


class TestCropMismappingMatrix(unittest.TestCase):
    """Rigorous collision testing for pairwise confused agricultural crops."""

    def test_01_ragi_vs_paddy(self):
        """Pair 1: Ragi (ರಾಗಿ) vs Paddy (ಭತ್ತ / ಅಕ್ಕಿ)."""
        self.assertEqual(normalize_crop_name("ragi"), "ragi")
        self.assertEqual(normalize_crop_name("ರಾಗಿ"), "ragi")
        self.assertEqual(normalize_crop_name("ರಾಗಿಯ"), "ragi")

        self.assertEqual(normalize_crop_name("paddy"), "paddy")
        self.assertEqual(normalize_crop_name("rice"), "paddy")
        self.assertEqual(normalize_crop_name("ಭತ್ತ"), "paddy")
        self.assertEqual(normalize_crop_name("ಅಕ್ಕಿ"), "paddy")
        self.assertEqual(normalize_crop_name("ಭತ್ತದ"), "paddy")

        self.assertNotEqual(normalize_crop_name("ರಾಗಿ"), normalize_crop_name("ಭತ್ತ"))

    def test_02_maize_vs_jowar(self):
        """Pair 2: Maize (ಮೆಕ್ಕೆಜೋಳ) vs Jowar (ಜೋಳ / ಬಿಳಿಜೋಳ)."""
        self.assertEqual(normalize_crop_name("maize"), "maize")
        self.assertEqual(normalize_crop_name("corn"), "maize")
        self.assertEqual(normalize_crop_name("ಮೆಕ್ಕೆಜೋಳ"), "maize")
        self.assertEqual(normalize_crop_name("ಮೆಕ್ಕೆ ಜೋಳ"), "maize")

        self.assertEqual(normalize_crop_name("jowar"), "jowar")
        self.assertEqual(normalize_crop_name("sorghum"), "jowar")
        self.assertEqual(normalize_crop_name("ಜೋಳ"), "jowar")
        self.assertEqual(normalize_crop_name("ಬಿಳಿಜೋಳ"), "jowar")

        # Ensure 'ಮೆಕ್ಕೆಜೋಳ' is NEVER partially matched to 'jowar'
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಮೆಕ್ಕೆಜೋಳ ಬೆಳೆಗೆ ರೋಗ ಬಂದಿದೆ"), "maize")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಜೋಳ ಬೆಳೆಗೆ ಸುಳಿ ನೊಣ ಬಂದಿದೆ"), "jowar")

    def test_03_jowar_vs_bajra(self):
        """Pair 3: Jowar (ಜೋಳ) vs Bajra (ಸಜ್ಜೆ)."""
        self.assertEqual(normalize_crop_name("jowar"), "jowar")
        self.assertEqual(normalize_crop_name("ಜೋಳ"), "jowar")
        self.assertEqual(normalize_crop_name("bajra"), "bajra")
        self.assertEqual(normalize_crop_name("pearl millet"), "bajra")
        self.assertEqual(normalize_crop_name("ಸಜ್ಜೆ"), "bajra")

        self.assertEqual(detect_crop_from_text("ಸಜ್ಜೆ ಬೆಳೆ ರೋಗ ನಿರ್ವಹಣೆ"), "bajra")
        self.assertEqual(detect_crop_from_text("ಜೋಳ ಬೆಳೆ ರೋಗ ನಿರ್ವಹಣೆ"), "jowar")

    def test_04_red_gram_vs_green_gram(self):
        """Pair 4: Red gram (ತೊಗರಿ) vs Green gram (ಹೆಸರು)."""
        self.assertEqual(normalize_crop_name("red gram"), "red_gram")
        self.assertEqual(normalize_crop_name("pigeon pea"), "red_gram")
        self.assertEqual(normalize_crop_name("tur"), "red_gram")
        self.assertEqual(normalize_crop_name("ತೊಗರಿ"), "red_gram")
        self.assertEqual(normalize_crop_name("ತೊಗರಿಕಾಳು"), "red_gram")

        self.assertEqual(normalize_crop_name("green gram"), "green_gram")
        self.assertEqual(normalize_crop_name("moong"), "green_gram")
        self.assertEqual(normalize_crop_name("mung bean"), "green_gram")
        self.assertEqual(normalize_crop_name("ಹೆಸರು"), "green_gram")
        self.assertEqual(normalize_crop_name("ಹೆಸರುಕಾಳು"), "green_gram")

        self.assertEqual(detect_crop_from_text("ತೊಗರಿ ಕಾಯಿ ಕೊರಕ ಹುಳು"), "red_gram")
        self.assertEqual(detect_crop_from_text("ಹೆಸರು ಬೆಳೆ ಹಳದಿ ಮೊಸಾಯಿಕ್"), "green_gram")

    def test_05_black_gram_vs_green_gram(self):
        """Pair 5: Black gram (ಉದ್ದು) vs Green gram (ಹೆಸರು)."""
        self.assertEqual(normalize_crop_name("black gram"), "black_gram")
        self.assertEqual(normalize_crop_name("urad"), "black_gram")
        self.assertEqual(normalize_crop_name("ಉದ್ದು"), "black_gram")
        self.assertEqual(normalize_crop_name("ಉದ್ದಿನಕಾಳು"), "black_gram")

        self.assertEqual(normalize_crop_name("green gram"), "green_gram")
        self.assertEqual(normalize_crop_name("ಹೆಸರು"), "green_gram")

        self.assertEqual(detect_crop_from_text("ಉದ್ದಿನ ಬೆಳೆ ರೋಗ"), "black_gram")
        self.assertEqual(detect_crop_from_text("ಹೆಸರಿನ ಬೆಳೆ ರೋಗ"), "green_gram")

    def test_06_chilli_vs_black_pepper(self):
        """Pair 6: Chilli (ಮೆಣಸಿನಕಾಯಿ / ಹಸಿಮೆಣಸು) vs Black pepper (ಕರಿಮೆಣಸು / ಕಾಳುಮೆಣಸು)."""
        self.assertEqual(normalize_crop_name("chilli"), "chilli")
        self.assertEqual(normalize_crop_name("red chilli"), "chilli")
        self.assertEqual(normalize_crop_name("green chilli"), "chilli")
        self.assertEqual(normalize_crop_name("ಮೆಣಸಿನಕಾಯಿ"), "chilli")
        self.assertEqual(normalize_crop_name("ಹಸಿಮೆಣಸಿನಕಾಯಿ"), "chilli")
        self.assertEqual(normalize_crop_name("ಬ್ಯಾಡಗಿ ಮೆಣಸಿನಕಾಯಿ"), "chilli")

        self.assertEqual(normalize_crop_name("black pepper"), "black_pepper")
        self.assertEqual(normalize_crop_name("ಕರಿಮೆಣಸು"), "black_pepper")
        self.assertEqual(normalize_crop_name("ಕಾಳುಮೆಣಸು"), "black_pepper")

        self.assertEqual(detect_crop_from_text("ಮೆಣಸಿನಕಾಯಿ ಎಲೆ ಮುದುರು ರೋಗ"), "chilli")
        self.assertEqual(detect_crop_from_text("ಕಾಳುಮೆಣಸು ದ್ರುತ ಸೊರಗು ರೋಗ"), "black_pepper")
        self.assertEqual(detect_crop_from_text("ಕರಿಮೆಣಸು ಕೊಳೆ ರೋಗ"), "black_pepper")

    def test_07_coriander_vs_cumin(self):
        """Pair 7: Coriander (ಕೊತ್ತಂಬರಿ) vs Cumin (ಜೀರಿಗೆ)."""
        self.assertEqual(normalize_crop_name("coriander"), "coriander")
        self.assertEqual(normalize_crop_name("ಕೊತ್ತಂಬರಿ"), "coriander")

        self.assertEqual(normalize_crop_name("cumin"), "cumin")
        self.assertEqual(normalize_crop_name("jeera"), "cumin")
        self.assertEqual(normalize_crop_name("ಜೀರಿಗೆ"), "cumin")

        self.assertEqual(get_karnataka_suitability("coriander"), "KARNATAKA_RELEVANT")
        self.assertEqual(get_karnataka_suitability("cumin"), "KARNATAKA_NOT_RECOMMENDED")

    def test_08_ginger_vs_turmeric(self):
        """Pair 8: Ginger (ಶುಂಠಿ) vs Turmeric (ಅರಿಶಿನ)."""
        self.assertEqual(normalize_crop_name("ginger"), "ginger")
        self.assertEqual(normalize_crop_name("ಶುಂಠಿ"), "ginger")
        self.assertEqual(normalize_crop_name("ಶುಂಠಿಯ"), "ginger")

        self.assertEqual(normalize_crop_name("turmeric"), "turmeric")
        self.assertEqual(normalize_crop_name("ಅರಿಶಿನ"), "turmeric")
        self.assertEqual(normalize_crop_name("ಅರಿಷಿಣ"), "turmeric")

        self.assertEqual(detect_crop_from_text("ಶುಂಠಿ ಗೆಡ್ಡೆ ಕೊಳೆ ರೋಗ"), "ginger")
        self.assertEqual(detect_crop_from_text("ಅರಿಶಿನ ಗೆಡ್ಡೆ ಕೊಳೆ ರೋಗ"), "turmeric")

    def test_09_garlic_vs_onion(self):
        """Pair 9: Garlic (ಬೆಳ್ಳುಳ್ಳಿ) vs Onion (ಈರುಳ್ಳಿ)."""
        self.assertEqual(normalize_crop_name("garlic"), "garlic")
        self.assertEqual(normalize_crop_name("ಬೆಳ್ಳುಳ್ಳಿ"), "garlic")

        self.assertEqual(normalize_crop_name("onion"), "onion")
        self.assertEqual(normalize_crop_name("ಈರುಳ್ಳಿ"), "onion")

        self.assertEqual(detect_crop_from_text("ಈರುಳ್ಳಿ ಪರ್ಪಲ್ ಬ್ಲಾಚ್ ರೋಗ"), "onion")
        self.assertEqual(detect_crop_from_text("ಬೆಳ್ಳುಳ್ಳಿ ಕೃಷಿ ಮಾಹಿತಿ"), "garlic")

    def test_10_brinjal_vs_tomato(self):
        """Pair 10: Brinjal (ಬದನೆಕಾಯಿ) vs Tomato (ಟೊಮ್ಯಾಟೊ)."""
        self.assertEqual(normalize_crop_name("brinjal"), "brinjal")
        self.assertEqual(normalize_crop_name("eggplant"), "brinjal")
        self.assertEqual(normalize_crop_name("ಬದನೆಕಾಯಿ"), "brinjal")
        self.assertEqual(normalize_crop_name("ಬದನೆ"), "brinjal")

        self.assertEqual(normalize_crop_name("tomato"), "tomato")
        self.assertEqual(normalize_crop_name("ಟೊಮ್ಯಾಟೊ"), "tomato")
        self.assertEqual(normalize_crop_name("ಟೊಮೇಟೊ"), "tomato")

        self.assertEqual(detect_crop_from_text("ಬದನೆಕಾಯಿ ಕಾಯಿ ಕೊರಕ ಹುಳು"), "brinjal")
        self.assertEqual(detect_crop_from_text("ಟೊಮ್ಯಾಟೊ ಎಲೆ ಹಳದಿ ರೋಗ"), "tomato")

    def test_11_watermelon_vs_muskmelon(self):
        """Pair 11: Watermelon (ಕಲ್ಲಂಗಡಿ) vs Muskmelon (ಕರಬೂಜ)."""
        self.assertEqual(normalize_crop_name("watermelon"), "watermelon")
        self.assertEqual(normalize_crop_name("ಕಲ್ಲಂಗಡಿ"), "watermelon")
        self.assertEqual(normalize_crop_name("ಕಲ್ಲಂಗಡಿಹಣ್ಣು"), "watermelon")

        self.assertEqual(normalize_crop_name("muskmelon"), "muskmelon")
        self.assertEqual(normalize_crop_name("cantaloupe"), "muskmelon")
        self.assertEqual(normalize_crop_name("ಕರಬೂಜ"), "muskmelon")

        self.assertEqual(get_crop_support_status("watermelon"), "supported")
        self.assertEqual(get_crop_support_status("muskmelon"), "recognized_not_supported")

        self.assertEqual(detect_crop_from_text("ಕಲ್ಲಂಗಡಿ ಬೆಳೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ"), "watermelon")
        self.assertEqual(detect_crop_from_text("ಕರಬೂಜ ಹಣ್ಣಿನ ಕೃಷಿ"), "muskmelon")

    def test_12_coffee_vs_cocoa(self):
        """Pair 12: Coffee (ಕಾಫಿ) vs Cocoa (ಕೋಕೋ)."""
        self.assertEqual(normalize_crop_name("coffee"), "coffee")
        self.assertEqual(normalize_crop_name("ಕಾಫಿ"), "coffee")

        self.assertEqual(normalize_crop_name("cocoa"), "cocoa")
        self.assertEqual(normalize_crop_name("cacao"), "cocoa")
        self.assertEqual(normalize_crop_name("ಕೋಕೋ"), "cocoa")

        self.assertEqual(get_crop_support_status("coffee"), "supported")
        self.assertEqual(get_crop_support_status("cocoa"), "recognized_not_supported")

    def test_13_coconut_vs_arecanut(self):
        """Pair 13: Coconut (ತೆಂಗು) vs Arecanut (ಅಡಿಕೆ)."""
        self.assertEqual(normalize_crop_name("coconut"), "coconut")
        self.assertEqual(normalize_crop_name("ತೆಂಗು"), "coconut")
        self.assertEqual(normalize_crop_name("ತೆಂಗಿನಕಾಯಿ"), "coconut")

        self.assertEqual(normalize_crop_name("arecanut"), "arecanut")
        self.assertEqual(normalize_crop_name("betel nut"), "arecanut")
        self.assertEqual(normalize_crop_name("ಅಡಿಕೆ"), "arecanut")

        self.assertEqual(detect_crop_from_text("ತೆಂಗಿನ ಮರದಲ್ಲಿ ಸುಳಿ ಕೊಳೆ"), "coconut")
        self.assertEqual(detect_crop_from_text("ಅಡಿಕೆ ಮರದಲ್ಲಿ ಮಹಾಳಿ ಕೊಳೆ ರೋಗ"), "arecanut")

    def test_14_apple_vs_custard_apple(self):
        """Pair 14: Apple (ಸೇಬು) vs Custard apple (ಸೀತಾಫಲ)."""
        self.assertEqual(normalize_crop_name("apple"), "apple")
        self.assertEqual(normalize_crop_name("ಸೇಬು"), "apple")
        self.assertEqual(normalize_crop_name("ಸೇಬುಹಣ್ಣು"), "apple")

        self.assertEqual(normalize_crop_name("custard apple"), "custard_apple")
        self.assertEqual(normalize_crop_name("sitaphal"), "custard_apple")
        self.assertEqual(normalize_crop_name("ಸೀತಾಫಲ"), "custard_apple")

        self.assertEqual(get_karnataka_suitability("apple"), "KARNATAKA_NOT_RECOMMENDED")
        self.assertEqual(get_karnataka_suitability("custard_apple"), "KARNATAKA_RELEVANT")

        # Longest match must prioritize custard apple over apple
        self.assertEqual(detect_crop_from_text("ಸೀತಾಫಲ ಹಣ್ಣಿನ ಬೆಳೆ"), "custard_apple")
        self.assertEqual(detect_crop_from_text("ಸೇಬು ಹಣ್ಣು ಕರ್ನಾಟಕದಲ್ಲಿ"), "apple")
        self.assertEqual(detect_crop_from_text("custard apple farming"), "custard_apple")
        self.assertEqual(detect_crop_from_text("apple farming in karnataka"), "apple")

    def test_15_sweet_potato_vs_potato(self):
        """Pair 15: Sweet Potato (ಗೆಣಸು) vs Potato (ಆಲೂಗಡ್ಡೆ)."""
        self.assertEqual(normalize_crop_name("sweet potato"), "sweet_potato")
        self.assertEqual(normalize_crop_name("ಗೆಣಸು"), "sweet_potato")

        self.assertEqual(normalize_crop_name("potato"), "potato")
        self.assertEqual(normalize_crop_name("ಆಲೂಗಡ್ಡೆ"), "potato")

        self.assertEqual(get_crop_support_status("potato"), "supported")
        self.assertEqual(get_crop_support_status("sweet_potato"), "recognized_not_supported")


if __name__ == "__main__":
    unittest.main()
