#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SEGMENTS = 12
SWITCH_INTERVAL = 18
DEFAULT_OUTPUT = "example/interactive_50cases_12prompts_18latent.jsonl"

BEATS = [
    "the shot opens with the subject holding position beside the key prop, letting the environment settle",
    "the subject begins a small deliberate motion toward the key prop, with natural body movement",
    "the subject reaches the key prop and makes first contact, keeping the same camera angle and lighting",
    "the subject turns slightly while continuing the action, preserving the same identity, outfit, and scene layout",
    "the main action becomes clearer and more energetic, with the key prop remaining visually important",
    "the subject continues the action in a smooth rhythm, adding a modest change in pose or direction",
    "a small environmental detail changes around the subject, but the location and continuity remain stable",
    "the subject moves around the key prop and re-centers in frame, with consistent scale and perspective",
    "the action reaches its most dynamic moment while staying realistic and physically coherent",
    "the subject slows down and regains a steady posture, still interacting with the same prop",
    "the subject gives a final expressive gesture that resolves the action without changing the scene",
    "the subject settles into a calm ending pose, leaving the frame ready for a seamless long-video continuation",
]

CASES = [
    ("case001_rainy_crosswalk", "a young woman in a yellow raincoat", "a wet neon city crosswalk at night", "a transparent umbrella", "opening the umbrella and crossing the street", "cinematic realistic medium shot"),
    ("case002_desert_astronaut", "a lone astronaut in a white EVA suit", "a quiet red desert plain under a pale blue sky", "a compact rover", "inspecting the rover antenna", "realistic science fiction wide shot"),
    ("case003_mountain_cyclist", "a cyclist in a red jacket and black helmet", "a misty alpine ridge trail", "a matte black mountain bike", "riding along the ridge", "documentary outdoor tracking shot"),
    ("case004_sushi_chef", "a focused sushi chef in a white jacket", "a small wooden sushi counter with warm lantern light", "a bamboo sushi mat", "preparing a roll", "realistic close medium shot"),
    ("case005_jazz_saxophonist", "a saxophonist in a dark suit", "a smoky jazz club with blue stage light", "a brass saxophone", "performing a solo", "cinematic low light medium shot"),
    ("case006_robot_lab", "a friendly humanoid robot with white panels", "a clean robotics lab with glass walls", "a small metal cube", "learning to stack the cube", "realistic technology demo shot"),
    ("case007_ballet_studio", "a ballerina in a pale lavender practice dress", "a sunlit rehearsal studio with mirrors", "a wooden barre", "practicing a turn sequence", "graceful realistic wide shot"),
    ("case008_market_vendor", "an elderly fruit vendor in a blue apron", "a busy morning street market", "a crate of oranges", "arranging fruit for customers", "natural documentary medium shot"),
    ("case009_arctic_researcher", "a polar researcher in a red parka", "a snowy Arctic field camp", "a weather sensor pole", "checking the instrument readings", "realistic cold daylight wide shot"),
    ("case010_subway_musician", "a young street musician in a denim jacket", "an underground subway platform", "an acoustic guitar", "playing for passing commuters", "urban handheld medium shot"),
    ("case011_forest_ranger", "a forest ranger in a green uniform", "a dense pine forest trail after rain", "a trail map", "marking a route on the map", "realistic nature documentary shot"),
    ("case012_kite_beach", "a child in a striped shirt", "a breezy beach at golden hour", "a red diamond kite", "launching the kite into the air", "warm realistic wide shot"),
    ("case013_pottery_artist", "a ceramic artist in a linen apron", "a quiet pottery studio with clay shelves", "a spinning pottery wheel", "shaping a clay bowl", "tactile realistic close medium shot"),
    ("case014_firefighter_training", "a firefighter in full turnout gear", "a controlled training yard with smoke", "a coiled fire hose", "advancing the hose line", "realistic action medium shot"),
    ("case015_library_student", "a university student in a green sweater", "a grand old library reading room", "a stack of open books", "finding a note in the books", "quiet cinematic medium shot"),
    ("case016_rooftop_cook", "a home cook in a striped shirt", "a rooftop kitchen overlooking a city skyline", "a sizzling pan", "tossing vegetables in the pan", "realistic evening medium shot"),
    ("case017_ceremony_drummer", "a ceremonial drummer in embroidered clothing", "a village square decorated with flags", "a large hand drum", "building a steady rhythm", "colorful cultural documentary shot"),
    ("case018_horse_trainer", "a horse trainer in a brown vest", "a sunlit riding arena", "a chestnut horse", "guiding the horse in a slow circle", "realistic ranch medium wide shot"),
    ("case019_tea_master", "a tea master in a simple gray robe", "a tranquil tea room with paper screens", "a clay teapot", "pouring tea with precision", "minimal realistic close shot"),
    ("case020_street_dancer", "a street dancer in a white hoodie", "a painted basketball court at dusk", "a portable speaker", "performing footwork", "energetic realistic wide shot"),
    ("case021_deep_sea_pilot", "a submersible pilot in an orange suit", "a compact deep sea control cabin", "a glowing sonar display", "navigating past a reef", "realistic claustrophobic medium shot"),
    ("case022_bakery_morning", "a baker in a flour-dusted apron", "a bright bakery kitchen before opening", "a tray of croissants", "sliding pastries into the oven", "warm realistic medium shot"),
    ("case023_wedding_photographer", "a wedding photographer in a black shirt", "a garden wedding aisle with white flowers", "a professional camera", "framing the couple", "soft realistic event shot"),
    ("case024_snowboarder", "a snowboarder in a teal jacket", "a snowy mountain slope under clear sun", "a blue snowboard", "carving through fresh snow", "dynamic outdoor tracking shot"),
    ("case025_violin_maker", "an artisan violin maker with round glasses", "a quiet workshop lined with wood pieces", "a half-finished violin", "polishing the varnish", "realistic workshop close shot"),
    ("case026_museum_guard", "a museum guard in a navy uniform", "a modern art gallery after hours", "a small flashlight", "checking a painting", "moody realistic medium shot"),
    ("case027_carnival_magician", "a carnival magician in a burgundy coat", "a colorful fairground booth at night", "a deck of cards", "performing a card reveal", "vibrant realistic medium shot"),
    ("case028_space_greenhouse", "a botanist in a light gray jumpsuit", "a futuristic orbital greenhouse", "a tray of glowing seedlings", "watering the plants", "clean science fiction medium shot"),
    ("case029_coffee_roaster", "a coffee roaster in a black apron", "a small roastery with copper equipment", "a scoop of roasted beans", "sampling the aroma", "warm realistic close medium shot"),
    ("case030_lighthouse_keeper", "a lighthouse keeper in a navy coat", "a stormy cliffside lighthouse balcony", "a brass lantern", "shielding the lantern from wind", "dramatic realistic wide shot"),
    ("case031_origami_child", "a child in a red sweater", "a cozy classroom craft table", "a sheet of blue paper", "folding an origami crane", "gentle realistic medium shot"),
    ("case032_monk_courtyard", "a monk in saffron robes", "a quiet stone monastery courtyard", "a wooden broom", "sweeping fallen leaves", "serene realistic wide shot"),
    ("case033_fashion_runway", "a model in a silver jacket", "a minimalist fashion runway with white light", "a reflective clutch", "walking and posing", "clean editorial medium shot"),
    ("case034_blacksmith", "a blacksmith in a leather apron", "a forge glowing with orange fire", "a heated iron rod", "hammering the metal", "gritty realistic close medium shot"),
    ("case035_farmer_greenhouse", "a farmer in a plaid shirt", "a humid tomato greenhouse", "a basket of ripe tomatoes", "harvesting carefully", "natural realistic medium shot"),
    ("case036_news_anchor", "a news anchor in a navy blazer", "a modern television studio", "a stack of cue cards", "preparing for a live segment", "polished realistic studio shot"),
    ("case037_ice_sculptor", "an ice sculptor in a padded black vest", "a winter festival carving tent", "a clear ice block", "chiseling a shape", "crisp realistic medium shot"),
    ("case038_tai_chi_park", "an elderly tai chi practitioner in white clothes", "a quiet city park at sunrise", "a stone path", "flowing through a slow form", "peaceful realistic wide shot"),
    ("case039_ship_captain", "a ship captain in a dark pea coat", "a rain-speckled bridge of a cargo ship", "a brass compass", "checking the heading", "realistic maritime medium shot"),
    ("case040_glassblower", "a glassblower in safety glasses", "a hot glass studio with furnace light", "a glowing glass gather", "turning the blowpipe", "warm realistic close shot"),
    ("case041_archery_range", "an archer in a forest green jacket", "an outdoor archery range", "a wooden bow", "drawing and releasing an arrow", "realistic sports medium shot"),
    ("case042_coral_diver", "a scuba diver in a black wetsuit", "a clear tropical coral reef", "an underwater camera", "filming reef fish", "underwater realistic wide shot"),
    ("case043_mechanic_garage", "a mechanic in a gray work shirt", "a tidy auto garage", "a red toolbox", "repairing an engine part", "realistic workshop medium shot"),
    ("case044_calligrapher", "a calligrapher in a cream cardigan", "a quiet desk beside a large window", "a black ink brush", "writing a flowing character", "calm realistic close shot"),
    ("case045_martial_artist", "a martial artist in a white uniform", "a traditional wooden dojo", "a practice staff", "moving through a staff form", "disciplined realistic wide shot"),
    ("case046_train_conductor", "a train conductor in a blue cap", "a vintage railway platform", "a silver pocket watch", "signaling departure", "nostalgic realistic medium shot"),
    ("case047_puppet_theater", "a puppeteer in a black turtleneck", "a small theater with red curtains", "a wooden marionette", "bringing the puppet to life", "theatrical realistic medium shot"),
    ("case048_graffiti_artist", "a graffiti artist in a purple jacket", "a legal mural wall under daylight", "a spray paint can", "adding a final color layer", "urban realistic medium wide shot"),
    ("case049_paramedic", "a paramedic in a bright yellow jacket", "an ambulance bay in early morning light", "a medical kit", "checking emergency supplies", "realistic documentary medium shot"),
    ("case050_desert_painter", "a landscape painter in a wide-brim hat", "a quiet desert overlook at sunset", "a wooden easel", "painting the distant mesas", "warm realistic wide shot"),
]


def build_prompts(case):
    case_id, subject, setting, prop, action, style = case
    prompts = []
    for index, beat in enumerate(BEATS, start=1):
        prompt = (
            f"{style}. A continuous long video segment {index:02d} of 12: "
            f"{subject} in {setting}, centered around {prop}, {action}. "
            f"In this segment, {beat}. Keep the same subject identity, outfit, "
            f"location, prop appearance, camera style, lighting, and scale. "
            f"Smooth realistic motion, coherent temporal continuity, no text, no watermark."
        )
        prompts.append(prompt)
    return {
        "case_id": case_id,
        "latent_switch_interval": SWITCH_INTERVAL,
        "prompts": prompts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = [build_prompts(case) for case in CASES]
    assert len(records) == 50
    assert all(len(record["prompts"]) == SEGMENTS for record in records)

    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(records)} cases x {SEGMENTS} prompts to {output}")
    print(
        "Switch frame indices:",
        ", ".join(str(SWITCH_INTERVAL * i) for i in range(1, SEGMENTS)),
    )


if __name__ == "__main__":
    main()
