import re
import base64
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from PIL import Image, ExifTags
import torch
from io import BytesIO
from enum import Enum
from typing import Optional, Any,  Union
import requests
import time
import numpy as np
from PIL.ExifTags import TAGS
from PIL import Image, ImageOps, TiffImagePlugin, UnidentifiedImageError
import openai
from openai import OpenAI
import io
import traceback
import concurrent.futures
from PIL import Image
from tqdm import tqdm
from PIL import Image, ExifTags



target_resolutions = [
    (640, 1632),  # 640 * 1632 = 1044480
    (704, 1472),  # 704 * 1472 = 1036288
    (768, 1360),  # 768 * 1360 = 1044480
    (832, 1248),  # 832 * 1248 = 1038336
    (896, 1152),
    (960, 1088),  # 960 * 1088 = 1044480
    (992, 1056),  # 992 * 1056 = 1047552
    (1024, 1024),  # 1024 * 1024 = 1048576
    (1056, 992),  # 1056 * 992 = 1047552
    (1088, 960),  # 1088 * 960 = 1044480
    (1152, 896),
    (1248, 832),  # 1248 * 832 = 1038336
    (1360, 768),  # 1360 * 768 = 1044480
    (1472, 704),  # 1472 * 704 = 1036288
    (1632, 640),  # 1632 * 640 = 1044480
    # (768, 1360),   # 768 * 1360 = 1044480
    # (1472, 704),   # 1472 * 704 = 1036288
    # (1024, 1024),  # 1024 * 1024 = 1048576
]


# 该函数将根据图像的EXIF方向旋转图像
def apply_exif_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()

        if exif is not None:
            exif = dict(exif.items())
            orientation_value = exif.get(orientation)

            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        # cases: image don't have getexif
        pass

    return image
def process_image(base64_image):

    try:

        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data))

        img = apply_exif_orientation(img)  # Apply the EXIF orientation

        # Convert to 'RGB' if it is 'RGBA' or any other mode
        img = img.convert('RGB')

        # 计算原图像的宽高比
        original_aspect_ratio = img.width / img.height

        # 找到最接近原图像宽高比的目标分辨率
        target_resolution = min(target_resolutions, key=lambda res: abs(original_aspect_ratio - res[0] / res[1]))

        # 计算新的维度
        if img.width / target_resolution[0] < img.height / target_resolution[1]:
            new_width = target_resolution[0]
            new_height = int(img.height * target_resolution[0] / img.width)
        else:
            new_height = target_resolution[1]
            new_width = int(img.width * target_resolution[1] / img.height)

        # 等比缩放图像
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # 计算裁剪的区域
        left = int((img.width - target_resolution[0]) / 2)
        top = int((img.height - target_resolution[1]) / 2)
        right = int((img.width + target_resolution[0]) / 2)
        bottom = int((img.height + target_resolution[1]) / 2)

        # 裁剪图像
        img = img.crop((left, top, right, bottom))

        # 转换并保存图像为JPG格式
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_image

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return None

# 剔除caption中所有颜色和发型
def remove_color_words(caption):
    color_words = ["golden", 'Black', 'Navy Blue', 'Dark Blue', 'Blue', 'Stratos', 'Swamp', 'Resolution Blue', 'Deep Fir', 'Burnham', 'International Klein Blue', 'Prussian Blue', 'Midnight Blue', 'Smalt', 'Deep Teal', 'Cyprus', 'Kaitoke Green', 'Cobalt', 'Crusoe', 'Sherpa Blue', 'Endeavour', 'Camarone', 'Science Blue', 'Blue Ribbon', 'Tropical Rain Forest', 'Allports', 'Deep Cerulean', 'Lochmara', 'Azure Radiance', 'Teal', 'Bondi Blue', 'Pacific Blue', 'Persian Green', 'Jade', 'Caribbean Green', "Robin's Egg Blue", 'Green', 'Spring Green', 'Cyan / Aqua', 'Blue Charcoal', 'Midnight', 'Holly', 'Daintree', 'Cardin Green', 'County Green', 'Astronaut Blue', 'Regal Blue', 'Aqua Deep', 'Orient', 'Blue Stone', 'Fun Green', 'Pine Green', 'Blue Lagoon', 'Deep Sea', 'Green Haze', 'English Holly', 'Sherwood Green', 'Congress Blue', 'Evening Sea', 'Bahama Blue', 'Observatory', 'Cerulean', 'Tangaroa', 'Green Vogue', 'Mosque', 'Midnight Moss', 'Black Pearl', 'Blue Whale', 'Zuccini', 'Teal Blue', 'Deep Cove', 'Gulf Blue', 'Venice Blue', 'Watercourse', 'Catalina Blue', 'Tiber', 'Gossamer', 'Niagara', 'Tarawera', 'Jaguar', 'Black Bean', 'Deep Sapphire', 'Elf Green', 'Bright Turquoise', 'Downriver', 'Palm Green', 'Madison', 'Bottle Green', 'Deep Sea Green', 'Salem', 'Black Russian', 'Dark Fern', 'Japanese Laurel', 'Atoll', 'Cod Gray', 'Marshland', 'Gordons Green', 'Black Forest', 'San Felix', 'Malachite', 'Ebony', 'Woodsmoke', 'Racing Green', 'Surfie Green', 'Blue Chill', 'Black Rock', 'Bunker', 'Aztec', 'Bush', 'Cinder', 'Firefly', 'Torea Bay', 'Vulcan', 'Green Waterloo', 'Eden', 'Arapawa', 'Ultramarine', 'Elephant', 'Jewel', 'Diesel', 'Asphalt', 'Blue Zodiac', 'Parsley', 'Nero', 'Tory Blue', 'Bunting', 'Denim', 'Genoa', 'Mirage', 'Hunter Green', 'Big Stone', 'Celtic', 'Timber Green', 'Gable Green', 'Pine Tree', 'Chathams Blue', 'Deep Forest Green', 'Blumine', 'Palm Leaf', 'Nile Blue', 'Fun Blue', 'Lucky Point', 'Mountain Meadow', 'Tolopea', 'Haiti', 'Deep Koamaru', 'Acadia', 'Seaweed', 'Biscay', 'Matisse', 'Crowshead', 'Rangoon Green', 'Persian Blue', 'Everglade', 'Elm', 'Green Pea', 'Creole', 'Karaka', 'El Paso', 'Cello', 'Te Papa Green', 'Dodger Blue', 'Eastern Blue', 'Night Rider', 'Java', 'Jacksons Purple', 'Cloud Burst', 'Blue Dianne', 'Eternity', 'Deep Blue', 'Forest Green', 'Mallard', 'Violet', 'Kilamanjaro', 'Log Cabin', 'Black Olive', 'Green House', 'Graphite', 'Cannon Black', 'Port Gore', 'Shark', 'Green Kelp', 'Curious Blue', 'Paua', 'Paris M', 'Wood Bark', 'Gondola', 'Steel Gray', 'Ebony Clay', 'Bay of Many', 'Plantation', 'Eucalyptus', 'Oil', 'Astronaut', 'Mariner', 'Violent Violet', 'Bastille', 'Zeus', 'Charade', 'Jelly Bean', 'Jungle Green', 'Cherry Pie', 'Coffee Bean', 'Baltic Sea', 'Turtle Green', 'Cerulean Blue', 'Sepia Black', 'Valhalla', 'Heavy Metal', 'Blue Gem', 'Revolver', 'Bleached Cedar', 'Lochinvar', 'Mikado', 'Outer Space', 'St Tropaz', 'Jacaranda', 'Jacko Bean', 'Rangitoto', 'Rhino', 'Sea Green', 'Scooter', 'Onion', 'Governor Bay', 'Sapphire', 'Spectra', 'Casal', 'Melanzane', 'Cocoa Brown', 'Woodrush', 'San Juan', 'Turquoise', 'Eclipse', 'Pickled Bluewood', 'Azure', 'Calypso', 'Paradiso', 'Persian Indigo', 'Blackcurrant', 'Mine Shaft', 'Stromboli', 'Bilbao', 'Astral', 'Christalle', 'Thunder', 'Shamrock', 'Tamarind', 'Mardi Gras', 'Valentino', 'Jagger', 'Tuna', 'Chambray', 'Martinique', 'Tuatara', 'Waiouru', 'Ming', 'La Palma', 'Chocolate', 'Clinker', 'Brown Tumbleweed', 'Birch', 'Oracle', 'Blue Diamond', 'Grape', 'Dune', 'Oxford Blue', 'Clover', 'Limed Spruce', 'Dell', 'Toledo', 'Sambuca', 'Jacarta', 'William', 'Killarney', 'Keppel', 'Temptress', 'Aubergine', 'Jon', 'Treehouse', 'Amazon', 'Boston Blue', 'Windsor', 'Rebel', 'Meteorite', 'Dark Ebony', 'Camouflage', 'Bright Gray', 'Cape Cod', 'Lunar Green', 'Bean  ', 'Bistre', 'Goblin', 'Kingfisher Daisy', 'Cedar', 'English Walnut', 'Black Marlin', 'Ship Gray', 'Pelorous', 'Bronze', 'Cola', 'Madras', 'Minsk', 'Cabbage Pont', 'Tom Thumb', 'Mineral Green', 'Puerto Rico', 'Harlequin', 'Brown Pod', 'Cork', 'Masala', 'Thatch Green', 'Fiord', 'Viridian', 'Chateau Green', 'Ripe Plum', 'Paco', 'Deep Oak', 'Merlin', 'Gun Powder', 'East Bay', 'Royal Blue', 'Ocean Green', 'Burnt Maroon', 'Lisbon Brown', 'Faded Jade', 'Scarlet Gum', 'Iroko', 'Armadillo', 'River Bed', 'Green Leaf', 'Barossa', 'Morocco Brown', 'Mako', 'Kelp', 'San Marino', 'Picton Blue', 'Loulou', 'Crater Brown', 'Gray Asparagus', 'Steel Blue', 'Rustic Red', 'Bulgarian Rose', 'Clairvoyant', 'Cocoa Bean', 'Woody Brown', 'Taupe', 'Van Cleef', 'Brown Derby', 'Metallic Bronze', 'Verdun Green', 'Blue Bayoux', 'Bismark', 'Bracken', 'Deep Bronze', 'Mondo', 'Tundora', 'Gravel', 'Trout', 'Pigment Indigo', 'Nandor', 'Saddle', 'Abbey', 'Blackberry', 'Cab Sav', 'Indian Tan', 'Cowboy', 'Livid Brown', 'Rock', 'Punga', 'Bronzetone', 'Woodland', 'Mahogany', 'Bossanova', 'Matterhorn', 'Bronze Olive', 'Mulled Wine', 'Axolotl', 'Wedgewood', 'Shakespeare', 'Honey Flower', 'Daisy Bush', 'Indigo', 'Fern Green', 'Fruit Salad', 'Apple', 'Mortar', 'Kashmir Blue', 'Cutty Sark', 'Emerald', 'Emperor', 'Chalet Green', 'Como', 'Smalt Blue', 'Castro', 'Maroon Oak', 'Gigas', 'Voodoo', 'Victoria', 'Hippie Green', 'Heath', 'Judge Gray', 'Fuscous Gray', 'Vida Loca', 'Cioccolato', 'Saratoga', 'Finlandia', 'Havelock Blue', 'Fountain Blue', 'Spring Leaves', 'Saddle Brown', 'Scarpa Flow', 'Cactus', 'Hippie Blue', 'Wine Berry', 'Brown Bramble', 'Congo Brown', 'Millbrook', 'Waikawa Gray', 'Horizon', 'Jambalaya', 'Bordeaux', 'Mulberry Wood', 'Carnaby Tan', 'Comet', 'Redwood', 'Don Juan', 'Chicago', 'Verdigris', 'Dingley', 'Breaker Bay', 'Kabul', 'Hemlock', 'Irish Coffee', 'Mid Gray', 'Shuttle Gray', 'Aqua Forest', 'Tradewind', 'Horses Neck', 'Smoky', 'Corduroy', 'Danube', 'Espresso', 'Eggplant', 'Costa Del Sol', 'Glade Green', 'Buccaneer', 'Quincy', 'Butterfly Bush', 'West Coast', 'Finch', 'Patina', 'Fern', 'Blue Violet', 'Dolphin', 'Storm Dust', 'Siam', 'Nevada', 'Cornflower Blue', 'Viking', 'Rosewood', 'Cherrywood', 'Purple Heart', 'Fern Frond', 'Willow Grove', 'Hoki', 'Pompadour', 'Purple', 'Tyrian Purple', 'Dark Tan', 'Silver Tree', 'Bright Green', "Screamin' Green", 'Black Rose', 'Scampi', 'Ironside Gray', 'Viridian Green', 'Christi', 'Nutmeg Wood Finish', 'Zambezi', 'Salt Box', 'Tawny Port', 'Finn', 'Scorpion', 'Lynch', 'Spice', 'Himalaya', 'Soya Bean', 'Hairy Heath', 'Royal Purple', 'Shingle Fawn', 'Dorado', 'Bermuda Gray', 'Olive Drab', 'Eminence', 'Turquoise Blue', 'Lonestar', 'Pine Cone', 'Dove Gray', 'Juniper', 'Gothic', 'Red Oxide', 'Moccaccino', 'Pickled Bean', 'Dallas', 'Kokoda', 'Pale Sky', 'Cafe Royale', 'Flint', 'Highland', 'Limeade', 'Downy', 'Persian Plum', 'Sepia', 'Antique Bronze', 'Ferra', 'Coffee', 'Slate Gray', 'Cedar Wood Finish', 'Metallic Copper', 'Affair', 'Studio', 'Tobacco Brown', 'Yellow Metal', 'Peat', 'Olivetone', 'Storm Gray', 'Sirocco', 'Aquamarine Blue', 'Venetian Red', 'Old Copper', 'Go Ben', 'Raven', 'Seance', 'Raw Umber', 'Kimberly', 'Crocodile', 'Crete', 'Xanadu', 'Spicy Mustard', 'Limed Ash', 'Rolling Stone', 'Blue Smoke', 'Laurel', 'Mantis', 'Russett', 'Deluge', 'Cosmic', 'Blue Marguerite', 'Lima', 'Sky Blue', 'Dark Burgundy', 'Crown of Thorns', 'Walnut', 'Pablo', 'Pacifika', 'Oxley', 'Pastel Green', 'Japanese Maple', 'Mocha', 'Peanut', 'Camouflage Green', 'Wasabi', 'Ship Cove', 'Sea Nymph', 'Roman Coffee', 'Old Lavender', 'Rum', 'Fedora', 'Sandstone', 'Spray', 'Siren', 'Fuchsia Blue', 'Boulder', 'Wild Blue Yonder', 'De York', 'Red Beech', 'Cinnamon', 'Yukon Gold', 'Tapa', 'Waterloo ', 'Flax Smoke', 'Amulet', 'Asparagus', 'Kenyan Copper', 'Pesto', 'Topaz', 'Concord', 'Jumbo', 'Trendy Green', 'Gumbo', 'Acapulco', 'Neptune', 'Pueblo', 'Bay Leaf', 'Malibu', 'Bermuda', 'Copper Canyon', 'Claret', 'Peru Tan', 'Falcon', 'Mobster', 'Moody Blue', 'Chartreuse', 'Aquamarine', 'Maroon', 'Rose Bud Cherry', 'Falu Red', 'Red Robin', 'Vivid Violet', 'Russet', 'Friar Gray', 'Olive', 'Gray', 'Gulf Stream', 'Glacier', 'Seagull', 'Nutmeg', 'Spicy Pink', 'Empress', 'Spanish Green', 'Sand Dune', 'Gunsmoke', 'Battleship Gray', 'Merlot', 'Shadow', 'Chelsea Cucumber', 'Monte Carlo', 'Plum', 'Granny Smith', 'Chetwode Blue', 'Bandicoot', 'Bali Hai', 'Half Baked', 'Red Devil', 'Lotus', 'Ironstone', 'Bull Shot', 'Rusty Nail', 'Bitter', 'Regent Gray', 'Disco', 'Americano', 'Hurricane', 'Oslo Gray', 'Sushi', 'Spicy Mix', 'Kumera', 'Suva Gray', 'Avocado', 'Camelot', 'Solid Pink', 'Cannon Pink', 'Makara', 'Burnt Umber', 'True V', 'Clay Creek', 'Monsoon', 'Stack', 'Jordy Blue', 'Electric Violet', 'Monarch', 'Corn Harvest', 'Olive Haze', 'Schooner', 'Natural Gray', 'Mantle', 'Portage', 'Envy', 'Cascade', 'Riptide', 'Cardinal Pink', 'Mule Fawn', 'Potters Clay', 'Trendy Pink', 'Paprika', 'Sanguine Brown', 'Tosca', 'Cement', 'Granite Green', 'Manatee', 'Polo Blue', 'Red Berry', 'Rope', 'Opium', 'Domino', 'Mamba', 'Nepal', 'Pohutukawa', 'El Salva', 'Korma', 'Squirrel', 'Vista Blue', 'Burgundy', 'Old Brick', 'Hemp', 'Almond Frost', 'Sycamore', 'Sangria', 'Cumin', 'Beaver', 'Stonewall', 'Venus', 'Medium Purple', 'Cornflower', 'Algae Green', 'Copper Rust', 'Arrowtown', 'Scarlett', 'Strikemaster', 'Mountain Mist', 'Carmine', 'Brown', 'Leather', "Purple Mountain's Majesty", 'Lavender Purple', 'Pewter', 'Summer Green', 'Au Chico', 'Wisteria', 'Atlantis', 'Vin Rouge', 'Lilac Bush', 'Bazaar', 'Hacienda', 'Pale Oyster', 'Mint Green', 'Fresh Eggplant', 'Violet Eggplant', 'Tamarillo', 'Totem Pole', 'Copper Rose', 'Amethyst', 'Mountbatten Pink', 'Blue Bell', 'Prairie Sand', 'Toast', 'Gurkha', 'Olivine', 'Shadow Green', 'Oregon', 'Lemon Grass', 'Stiletto', 'Hawaiian Tan', 'Gull Gray', 'Pistachio', 'Granny Smith Apple', 'Anakiwa', 'Chelsea Gem', 'Sepia Skin', 'Sage', 'Citron', 'Rock Blue', 'Morning Glory', 'Cognac', 'Reef Gold', 'Star Dust', 'Santas Gray', 'Sinbad', 'Feijoa', 'Tabasco', 'Buttered Rum', 'Hit Gray', 'Citrus', 'Aqua Island', 'Water Leaf', 'Flirt', 'Rouge', 'Cape Palliser', 'Gray Chateau', 'Edward', 'Pharlap', 'Amethyst Smoke', 'Blizzard Blue', 'Delta', 'Wistful', 'Green Smoke', 'Jazzberry Jam', 'Zorba', 'Bahia', 'Roof Terracotta', 'Paarl', 'Barley Corn', 'Donkey Brown', 'Dawn', 'Mexican Red', 'Luxor Gold', 'Rich Gold', 'Reno Sand', 'Coral Tree', 'Dusty Gray', 'Dull Lavender', 'Tallow', 'Bud', 'Locust', 'Norway', 'Chinook', 'Gray Olive', 'Aluminium', 'Cadet Blue', 'Schist', 'Tower Gray', 'Perano', 'Opal', 'Night Shadz', 'Fire', 'Muesli', 'Sandal', 'Shady Lady', 'Logan', 'Spun Pearl', 'Regent St Blue', 'Magic Mint', 'Lipstick', 'Royal Heath', 'Sandrift', 'Cold Purple', 'Bronco', 'Limed Oak', 'East Side', 'Lemon Ginger', 'Napa', 'Hillary', 'Cloudy', 'Silver Chalice', 'Swamp Green', 'Spring Rain', 'Conifer', 'Celadon', 'Mandalay', 'Casper', 'Moss Green', 'Padua', 'Green Yellow', 'Hippie Pink', 'Desert', 'Bouquet', 'Medium Carmine', 'Apple Blossom', 'Brown Rust', 'Driftwood', 'Alpine', 'Lucky', 'Martini', 'Bombay', 'Pigeon Post', 'Cadillac', 'Matrix', 'Tapestry', 'Mai Tai', 'Del Rio', 'Powder Blue', 'Inch Worm', 'Bright Red', 'Vesuvius', 'Pumpkin Skin', 'Santa Fe', 'Teak', 'Fringy Flower', 'Ice Cold', 'Shiraz', 'Biloba Flower', 'Tall Poppy', 'Fiery Orange', 'Hot Toddy', 'Taupe Gray', 'La Rioja', 'Well Read', 'Blush', 'Jungle Mist', 'Turkish Rose', 'Lavender', 'Mongoose', 'Olive Green', 'Jet Stream', 'Cruise', 'Hibiscus', 'Thatch', 'Heathered Gray', 'Eagle', 'Spindle', 'Gum Leaf', 'Rust', 'Muddy Waters', 'Sahara', 'Husk', 'Nobel', 'Heather', 'Madang', 'Milano Red', 'Copper', 'Gimblet', 'Green Spring', 'Celery', 'Sail', 'Chestnut', 'Crail', 'Marigold', 'Wild Willow', 'Rainee', 'Guardsman Red', 'Rock Spray', 'Bourbon', 'Pirate Gold', 'Nomad', 'Submarine', 'Charlotte', 'Medium Red Violet', 'Brandy Rose', 'Rio Grande', 'Surf', 'Powder Ash', 'Tuscany', 'Quicksand', 'Silk', 'Malta', 'Chatelle', 'Lavender Gray', 'French Gray', 'Clay Ash', 'Loblolly', 'French Pass', 'London Hue', 'Pink Swan', 'Fuego', 'Rose of Sharon', 'Tide', 'Blue Haze', 'Silver Sand', 'Key Lime Pie', 'Ziggurat', 'Lime', 'Thunderbird', 'Mojo', 'Old Rose', 'Silver', 'Pale Leaf', 'Pixie Green', 'Tia Maria', 'Fuchsia Pink', 'Buddha Gold', 'Bison Hide', 'Tea', 'Gray Suit', 'Sprout', 'Sulu', 'Indochine', 'Twine', 'Cotton Seed', 'Pumice', 'Jagged Ice', 'Maroon Flush', 'Indian Khaki', 'Pale Slate', 'Gray Nickel', 'Periwinkle Gray', 'Tiara', 'Tropical Blue', 'Cardinal', 'Fuzzy Wuzzy Brown', 'Orange Roughy', 'Mist Gray', 'Coriander', 'Mint Tulip', 'Mulberry', 'Nugget', 'Tussock', 'Sea Mist', 'Yellow Green', 'Brick Red', 'Contessa', 'Oriental Pink', 'Roti', 'Ash', 'Kangaroo', 'Las Palmas', 'Monza', 'Red Violet', 'Coral Reef', 'Melrose', 'Cloud', 'Ghost', 'Pine Glade', 'Botticelli', 'Antique Brass', 'Lilac', 'Hokey Pokey', 'Lily', 'Laser', 'Edgewater', 'Piper', 'Pizza', 'Light Wisteria', 'Rodeo Dust', 'Sundance', 'Earls Green', 'Silver Rust', 'Conch', 'Reef', 'Aero Blue', 'Flush Mahogany', 'Turmeric', 'Paris White', 'Bitter Lemon', 'Skeptic', 'Viola', 'Foggy Gray', 'Green Mist', 'Nebula', 'Persian Red', 'Burnt Orange', 'Ochre', 'Puce', 'Thistle Green', 'Periwinkle', 'Electric Lime', 'Tenn', 'Chestnut Rose', 'Brandy Punch', 'Onahau', 'Sorrell Brown', 'Cold Turkey', 'Yuma', 'Chino', 'Eunry', 'Old Gold', 'Tasman', 'Surf Crest', 'Humming Bird', 'Scandal', 'Red Stage', 'Hopbush', 'Meteor', 'Perfume', 'Prelude', 'Tea Green', 'Geebung', 'Vanilla', 'Soft Amber', 'Celeste', 'Mischka', 'Pear', 'Hot Cinnamon', 'Raw Sienna', 'Careys Pink', 'Tan', 'Deco', 'Blue Romance', 'Gossip', 'Sisal', 'Swirl', 'Charm', 'Clam Shell', 'Straw', 'Akaroa', 'Bird Flower', 'Iron', 'Geyser', 'Hawkes Blue', 'Grenadier', 'Can Can', 'Whiskey', 'Winter Hazel', 'Granny Apple', 'My Pink', 'Tacha', 'Moon Raker', 'Quill Gray', 'Snowy Mint', 'New York Pink', 'Pavlova', 'Fog', 'Valencia', 'Japonica', 'Thistle', 'Maverick', 'Foam', 'Cabaret', 'Burning Sand', 'Cameo', 'Timberwolf', 'Tana', 'Link Water', 'Mabel', 'Cerise', 'Flame Pea', 'Bamboo', 'Red Damask', 'Orchid', 'Copperfield', 'Golden Grass', 'Zanah', 'Iceberg', 'Oyster Bay', 'Cranberry', 'Petite Orchid', 'Di Serria', 'Alto', 'Frosted Mint', 'Crimson', 'Punch', 'Galliano', 'Blossom', 'Wattle', 'Westar', 'Moon Mist', 'Caper', 'Swans Down', 'Swiss Coffee', 'White Ice', 'Cerise Red', 'Roman', 'Tumbleweed', 'Gold Tips', 'Brandy', 'Wafer', 'Sapling', 'Barberry', 'Beryl Green', 'Pattens Blue', 'Heliotrope', 'Apache', 'Chenin', 'Lola', 'Willow Brook', 'Chartreuse Yellow', 'Mauve', 'Anzac', 'Harvest Gold', 'Calico', 'Baby Blue', 'Sunglo', 'Equator', 'Pink Flare', 'Periglacial Blue', 'Kidnapper', 'Tara', 'Mandy', 'Terracotta', 'Golden Bell', 'Shocking', 'Dixie', 'Light Orchid', 'Snuff', 'Mystic', 'Apple Green', 'Razzmatazz', 'Alizarin Crimson', 'Cinnabar', 'Cavern Pink', 'Peppermint', 'Mindaro', 'Deep Blush', 'Gamboge', 'Melanie', 'Twilight', 'Bone', 'Sunflower', 'Grain Brown', 'Zombie', 'Frostee', 'Snow Flurry', 'Amaranth', 'Zest', 'Dust Storm', 'Stark White', 'Hampton', 'Bon Jour', 'Mercury', 'Polar', 'Trinidad', 'Gold Sand', 'Cashmere', 'Double Spanish White', 'Satin Linen', 'Harp', 'Off Green', 'Hint of Green', 'Tranquil', 'Mango Tango', 'Christine', 'Tonys Pink', 'Kobi', 'Rose Fog', 'Corn', 'Putty', 'Gray Nurse', 'Lily White', 'Bubbles', 'Fire Bush', 'Shilo', 'Pearl Bush', 'Green White', 'Chrome White', 'Gin', 'Aqua Squeeze', 'Clementine', 'Burnt Sienna', 'Tahiti Gold', 'Oyster Pink', 'Confetti', 'Ebb', 'Ottoman', 'Clear Day', 'Carissma', 'Porsche', 'Tulip Tree', 'Rob Roy', 'Raffia', 'White Rock', 'Panache', 'Solitude', 'Aqua Spring', 'Dew', 'Apricot', 'Zinnwaldite', 'Fuel Yellow', 'Ronchi', 'French Lilac', 'Just Right', 'Wild Rice', 'Fall Green', 'Aths Special', 'Starship', 'Red Ribbon', 'Tango', 'Carrot Orange', 'Sea Pink', 'Tacao', 'Desert Sand', 'Pancho', 'Chamois', 'Primrose', 'Frost', 'Aqua Haze', 'Zumthor', 'Narvik', 'Honeysuckle', 'Lavender Magenta', 'Beauty Bush', 'Chalky', 'Almond', 'Flax', 'Bizarre', 'Double Colonial White', 'Cararra', 'Manz', 'Tahuna Sands', 'Athens Gray', 'Tusk', 'Loafer', 'Catskill White', 'Twilight Blue', 'Jonquil', 'Rice Flower', 'Jaffa', 'Gallery', 'Porcelain', 'Mauvelous', 'Golden Dream', 'Golden Sand', 'Buff', 'Prim', 'Khaki', 'Selago', 'Titan White', 'Alice Blue', 'Feta', 'Gold Drop', 'Wewak', 'Sahara Sand', 'Parchment', 'Blue Chalk', 'Mint Julep', 'Seashell', 'Saltpan', 'Tidal', 'Chiffon', 'Flamingo', 'Tangerine', 'Mandys Pink', 'Concrete', 'Black Squeeze', 'Pomegranate', 'Buttercup', 'New Orleans', 'Vanilla Ice', 'Sidecar', 'Dawn Pink', 'Wheatfield', 'Canary', 'Orinoco', 'Carla', 'Hollywood Cerise', 'Sandy brown', 'Saffron', 'Ripe Lemon', 'Janna', 'Pampas', 'Wild Sand', 'Zircon', 'Froly', 'Cream Can', 'Manhattan', 'Maize', 'Wheat', 'Sandwisp', 'Pot Pourri', 'Albescent White', 'Soft Peach', 'Ecru White', 'Beige', 'Golden Fizz', 'Australian Mint', 'French Rose', 'Brilliant Rose', 'Illusion', 'Merino', 'Black Haze', 'Spring Sun', 'Violet Red', 'Chilean Fire', 'Persian Pink', 'Rajah', 'Azalea', 'We Peep', 'Quarter Spanish White', 'Whisper', 'Snow Drift', 'Casablanca', 'Chantilly', 'Cherub', 'Marzipan', 'Energy Yellow', 'Givry', 'White Linen', 'Magnolia', 'Spring Wood', 'Coconut Cream', 'White Lilac', 'Desert Storm', 'Texas', 'Corn Field', 'Mimosa', 'Carnation', 'Saffron Mango', 'Carousel Pink', 'Dairy Cream', 'Portica', 'Amour', 'Rum Swizzle', 'Dolly', 'Sugar Cane', 'Ecstasy', 'Tan Hide', 'Corvette', 'Peach Yellow', 'Turbo', 'Astra', 'Champagne', 'Linen', 'Fantasy', 'Citrine White', 'Alabaster', 'Hint of Yellow', 'Milan', 'Brink Pink', 'Geraldine', 'Lavender Rose', 'Sea Buckthorn', 'Sun', 'Lavender Pink', 'Rose Bud', 'Cupid', 'Classic Rose', 'Apricot Peach', 'Banana Mania', 'Marigold Yellow', 'Festival', 'Sweet Corn', 'Candy Corn', 'Hint of Red', 'Shalimar', 'Shocking Pink', 'Tickle Me Pink', 'Tree Poppy', 'Lightning Yellow', 'Goldenrod', 'Candlelight', 'Cherokee', 'Double Pearl Lusta', 'Pearl Lusta', 'Vista White', 'Bianca', 'Moon Glow', 'China Ivory', 'Ceramic', 'Torch Red', 'Wild Watermelon', 'Crusta', 'Sorbus', 'Sweet Pink', 'Light Apricot', 'Pig Pink', 'Cinderella', 'Golden Glow', 'Lemon', 'Old Lace', 'Half Colonial White', 'Drover', 'Pale Prim', 'Cumulus', 'Persian Rose', 'Sunset Orange', 'Bittersweet', 'California', 'Yellow Sea', 'Melon', 'Bright Sun', 'Dandelion', 'Salomie', 'Cape Honey', 'Remy', 'Oasis', 'Bridesmaid', 'Beeswax', 'Bleach White', 'Pipi', 'Half Spanish White', 'Wisp Pink', 'Provincial Pink', 'Half Dutch White', 'Solitaire', 'White Pointer', 'Off Yellow', 'Orange White', 'Red', 'Rose', 'Purple Pizzazz', 'Magenta / Fuchsia', 'Scarlet', 'Wild Strawberry', 'Razzle Dazzle Rose', 'Radical Red', 'Red Orange', 'Coral Red', 'Vermilion', 'International Orange', 'Outrageous Orange', 'Blaze Orange', 'Pink Flamingo', 'Orange', 'Hot Pink', 'Persimmon', 'Blush Pink', 'Burning Orange', 'Pumpkin', 'Flamenco', 'Flush Orange', 'Coral', 'Salmon', 'Pizazz', 'West Side', 'Pink Salmon', 'Neon Carrot', 'Atomic Tangerine', 'Vivid Tangerine', 'Sunshade', 'Orange Peel', 'Mona Lisa', 'Web Orange', 'Carnation Pink', 'Hit Pink', 'Yellow Orange', 'Cornflower Lilac', 'Sundown', 'My Sin', 'Texas Rose', 'Cotton Candy', 'Macaroni and Cheese', 'Selective Yellow', 'Koromiko', 'Amber', 'Wax Flower', 'Pink', 'Your Pink', 'Supernova', 'Flesh', 'Sunglow', 'Golden Tainoi', 'Peach Orange', 'Chardonnay', 'Pastel Pink', 'Romantic', 'Grandis', 'Gold', 'School bus Yellow', 'Cosmos', 'Mustard', 'Peach Schnapps', 'Caramel', 'Tuft Bush', 'Watusi', 'Pink Lace', 'Navajo White', 'Frangipani', 'Pippin', 'Pale Rose', 'Negroni', 'Cream Brulee', 'Peach', 'Tequila', 'Kournikova', 'Sandy Beach', 'Karry', 'Broom', 'Colonial White', 'Derby', 'Vis Vis', 'Egg White', 'Papaya Whip', 'Fair Pink', 'Peach Cream', 'Lavender blush', 'Gorse', 'Buttermilk', 'Pink Lady', 'Forget Me Not', 'Tutu', 'Picasso', 'Chardon', 'Paris Daisy', 'Barley White', 'Egg Sour', 'Sazerac', 'Serenade', 'Chablis', 'Seashell Peach', 'Sauvignon', 'Milk Punch', 'Varden', 'Rose White', 'Baja White', 'Gin Fizz', 'Early Dawn', 'Lemon Chiffon', 'Bridal Heath', 'Scotch Mist', 'Soapstone', 'Witch Haze', 'Buttery White', 'Island Spice', 'Cream', 'Chilean Heath', 'Travertine', 'Orchid White', 'Quarter Pearl Lusta', 'Half and Half', 'Apricot White', 'Rice Cake', 'Black White', 'Romance', 'Yellow', 'Laser Lemon', 'Pale Canary', 'Portafino', 'Ivory', 'White']
    hairstyles = ["Afro", "Afro puffs", "Asymmetric cut", "Bald", "Bangs", "Beehive", "Big hair", "Blowout", "Bob cut",
                  "Bouffant", "Bowl cut", "Braid", "Brush, butch, burr cut", "Bun (odango)", "Bunches",
                  "Businessman cut", "Butterfly haircut", "Buzz cut", "Caesar cut", "Chignon", "Chonmage", "Comb over",
                  "Conk", "Cornrows", "Crew cut", "Crochet braids", "Croydon facelift", "Curly hair", "Curtained hair",
                  "Czupryna", "Devilock", "Dido flip", "Digital perm", "Dreadlocks", "Ducktail", "Edgar cut",
                  "Eton crop", "Extensions", "Fauxhawk", "Feathered hair", "Finger wave", "Flattop", "Fontange",
                  "French braid", "French twist", "Fringe", "Frosted tips", "Hair crimping", "Hair twists",
                  "High and tight", "Hime cut", "Historical Christian hairstyles", "Hi-top fade", "Induction cut",
                  "Ivy League, Harvard, Princeton cut", "Japanese women", "Jewfro", "Jheri curl", "Kinky hair",
                  "Kiss curl", "Laid edges", "Layered hair", "Liberty spikes", "Long hair", "Lob cut", "Lovelock",
                  "Marcelling", "Mod cut", "Mohawk", "Mullet", "Pageboy", "Part", "Payot", "Pigtail", "Pixie cut",
                  "Pompadour", "Ponytail", "Punch perm", "Professional cut", "Queue", "Quiff", "Rattail", "Razor cut",
                  "Regular haircut", "Ringlets", "Shag", "Shape-Up", "Shikha", "Shimada", "Short back and sides",
                  "Short brush cut", "Short hair", "Spiky hair", "Straight hair", "Standard haircut", "Step cut",
                  "Surfer hair", "Taper cut", "Temple fade", "Tonsure", "Updo", "Undercut", "Victory rolls", "Waves",
                  "Widow's peak", "Wings", "braided"]
    pattern = r"\b(" + "|".join(color_words) + r")\b"
    filtered_words = re.sub(pattern, "", caption, flags=re.IGNORECASE)
    filtered_words = [word.strip().replace(",", "") for word in filtered_words.split(",") if word.strip() != ""]
    filtered_words = [word for word in filtered_words if not any(hairstyle.lower() in word.lower() for hairstyle in hairstyles)]
    return ', '.join(filtered_words)

# 剔除caption中包含在exclude_words中的单词的函数
def remove_exclude_words(caption, exclude_words):
    exclude_words_list = [word.strip(" ,，") for word in exclude_words.split(",")]
    pattern = r'\b(?:{})\b'.format('|'.join(exclude_words_list))
    clean_caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
    clean_caption = re.sub(r'(, )+', ', ', clean_caption)
    return clean_caption.strip(', ')
    
#添加提示词
def add_words_to_caption(caption, add_words):
    # Add a comma after the caption and append the additional words
    updated_caption = caption + ', ' + add_words
    
    return updated_caption

    
# 为提示词增加权重
def add_weight_to_prompt(prompt, weight):
    prompt = prompt.rstrip(',')
    formatted_weight = "{:.2f}".format(weight)
    weighted_prompt = f"{prompt}={formatted_weight}"
    return f"({weighted_prompt})"

class DalleImage:

    @staticmethod
    def tensor_to_base64(tensor: torch.Tensor) -> str:
        try:
            """
            将 PyTorch 张量转换为 base64 编码的图像。
    
            注意：ComfyUI 提供的图像张量格式为 [N, H, W, C]。
            例如，形状为 torch.Size([1, 1024, 1024, 3])
    
            参数:
                tensor (torch.Tensor): 要转换的图像张量。
    
            返回:
                str: base64 编码的图像字符串。
            """
            # 将张量转换为 PIL 图像
            if tensor.ndim == 4:
                tensor = tensor.squeeze(0)  # 如果存在批量维度，则移除
            pil_image = Image.fromarray((tensor.numpy() * 255).astype('uint8'))

            # 将 PIL 图像保存到缓冲区
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")  # 可以根据需要更改为 JPEG 格式
            buffer.seek(0)

            # 将缓冲区编码为 base64
            base64_image = base64.b64encode(buffer.read()).decode('utf-8')

            return base64_image

        except Exception as e:
            print(f"Error in tensor_to_base64: {e}")
            traceback.print_exc()
            return None

# 初始化缓存变量
cached_result = None
cached_seed = None
cached_image = None

class GPTCaptioner:
    def __init__(self):
        self.saved_api_key = ''
        self.saved_exclude_words = ''

    #定义输入参数类型和默认值
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_url": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"max": 0xffffffffffffffff, "min": 1, "step": 1, "default": 1, "display": "number"}),
                "prompt_type": (["generic", "figure"], {"default": "generic"}),
                "enable_weight": ("BOOLEAN", {"default": False}),
                "weight" : ("FLOAT", {"max": 8.201, "min": 0.1, "step": 0.1, "display": "number", "round": 0.01, "default": 1}), 
                "exclude_words": ("STRING",
                                   {
                                       "default": "",
                                       "multiline": True, "dynamicPrompts": False
                                   }),
                "add_words": ("STRING",
                                   {
                                       "default": "",
                                       "multiline": True, "dynamicPrompts": False
                                   }),
                "image": ("IMAGE", {})
            }
        }

    #定义输出名称、类型、所属位置
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("prompt","full_prompt")
    FUNCTION = "sanmi"
    OUTPUT_NODE = False
    CATEGORY = "Sanmi Simple Nodes/GPT"

    # 对排除词进行处理
    @staticmethod
    def clean_response_text(text: str) -> str:
        try:
            cleaned_text = text.replace("，", ",").replace("。", "")
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
            return cleaned_text

        except Exception as e:
            print(f"Error in clean_response_text: {e}")
            traceback.print_exc()
            return None


    # 调用 OpenAI API，将图像和文本提示发送给 API 并获取生成的文本描述，处理可能出现的异常情况，并返回相应的结果或错误信息。
    @staticmethod
    def run_openai_api(image, api_key, api_url, exclude_words, seed, prompt_type, add_words, quality='auto', timeout=10):
        global cached_result, cached_seed, cached_image
        # 判断seed值和image值是否发生变化
        if cached_seed is not None and cached_image is not None and cached_seed == seed and cached_image == image:
            caption = cached_result
            caption = remove_exclude_words(caption, exclude_words)
            caption = add_words_to_caption(caption, add_words)
            return caption      
        
        generic_prompt = "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."
        figure_prompt = "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, composition, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, attire, actions, pose, expressions, accessories, makeup, composition type, etc.  Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. The final tag results should exclude the following tags: color, hair color, hairstyle, clothing color, wig, style, watermarks, signatures, text, logos, backgrounds, lighting, filters, styles. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."
        image_base64 = image
        
        if prompt_type == 'generic':
            prompt = generic_prompt
        elif prompt_type == 'figure':   
            prompt = figure_prompt
        
        
        
        data = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": f"{quality}"
                        }
                         }
                    ]
                }
            ],
            "max_tokens": 300
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 配置重试策略
        retries = Retry(total=5,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"])  # 更新参数名
        #处理可能发生的请求异常，并返回相应的错误消息。
        with requests.Session() as s:
            s.mount('https://', HTTPAdapter(max_retries=retries))

            try:
                response = s.post(api_url, headers=headers, json=data, timeout=timeout)
                response.raise_for_status()  # 如果请求失败，将抛出 HTTPError
            except requests.exceptions.HTTPError as errh:
                return f"HTTP Error: {errh}"
            except requests.exceptions.ConnectionError as errc:
                return f"Error Connecting: {errc}"
            except requests.exceptions.Timeout as errt:
                return f"Timeout Error: {errt}"
            except requests.exceptions.RequestException as err:
                return f"OOps: Something Else: {err}"

        try:
            response_data = response.json()

            if 'error' in response_data:
                return f"API error: {response_data['error']['message']}"

            caption = response_data["choices"][0]["message"]["content"]

            # 更新缓存变量
            cached_result = caption
            cached_seed = seed
            cached_image = image

            full_caption = caption
            # 剔除caption中所有颜色和发型
            if prompt_type == 'figure':
                caption = remove_color_words(caption)

            # 剔除caption中包含在exclude_words中的单词
            caption = remove_exclude_words(caption, exclude_words)
            
            # 增加提示词
            caption = add_words_to_caption(caption, add_words)

            return (caption,full_caption)
        except Exception as e:
            return (f"Failed to parse the API response: {e}\n{response.text}",None)


    # 根据用户输入的参数构建指令，并使用 GPT 模型进行请求，返回相应的结果。将之前的值进行转换
    def sanmi(self, api_key, api_url, exclude_words, image, seed, prompt_type, weight,enable_weight,add_words):
        try:

            # 初始化 prompt
            prompt = ""

            # 如果 image 是 torch.Tensor 类型，则将其转换为 base64 编码的图像
            if isinstance(image, torch.Tensor):
                image = DalleImage.tensor_to_base64(image)

            # 自动裁切base64 编码图像
            image = process_image(image)

            # 请求 prompt
            caption,full_caption = self.run_openai_api(image, api_key, api_url, exclude_words, seed, prompt_type,add_words)
            
            # 给 prompt 增加权重
            if enable_weight:
                caption = add_weight_to_prompt(caption, weight)

            return (caption,full_caption)
        except Exception as e:
            print(f"Error in sanmi: {e}")
            traceback.print_exc()
            return None

#定义功能和模块名称
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GPT4VCaptioner": GPTCaptioner,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT4VCaptioner": "GPT4V-Image-Captioner",
}







