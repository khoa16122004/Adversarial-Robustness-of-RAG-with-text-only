from retrieval import Retriever
from reader import Reader
from utils import set_seed_everything
import numpy as np

set_seed_everything(222520691)

retriever = Retriever("facebook/dpr-question_encoder-multiset-base", 
                      "facebook/dpr-ctx_encoder-multiset-base")

reader = Reader("Llama-7b")

question = "What is the fastest land animal?"
answer = "The cheetah."
documents = [
      "The cheetah is the fastest land animal, capable of reaching speeds up to 70 mph. It has a slender build and distinctive spotted coat. Cheetahs primarily hunt gazelles and other small antelopes in Africa.",
      "Lions are known as the king of the jungle despite living in grasslands. They live in social groups called prides. Male lions have distinctive manes that darken with age.",
      "Elephants are the largest land mammals on Earth. They have excellent memories and complex social structures. Elephants use their trunks for breathing, drinking, and grasping objects.",
      "Polar bears are the largest carnivorous land mammals. They are excellent swimmers and primarily hunt seals. Polar bears have black skin underneath their white fur.",
      "Giraffes are the tallest mammals in the world. Their long necks help them reach leaves high in trees. A giraffe's tongue can be up to 20 inches long.",
      "Dolphins are highly intelligent marine mammals. They communicate through clicks, whistles, and body language. Dolphins live in social groups called pods.",
      "Kangaroos are marsupials native to Australia. They move by hopping on their powerful hind legs. Female kangaroos carry their young in pouches.",
      "Penguins are flightless birds adapted for life in water. They have waterproof feathers and streamlined bodies. Most penguin species live in the Southern Hemisphere.",
      "Koalas are marsupials that primarily eat eucalyptus leaves. They sleep 18-22 hours per day to conserve energy. Koalas are found only in Australia.",
      "Tigers are the largest wild cats in the world. Each tiger has a unique stripe pattern like human fingerprints. Tigers are solitary hunters and excellent swimmers.",
      "Whales are the largest animals on Earth. Blue whales can grow up to 100 feet long. They communicate through complex songs that can travel for miles underwater.",
      "Gorillas are the largest primates and share 98% of human DNA. They live in family groups led by a dominant silverback male. Gorillas are primarily herbivorous despite their size.",
      "Pandas are bears known for their distinctive black and white coloring. They primarily eat bamboo and spend most of their time eating. Giant pandas are endangered with fewer than 2,000 remaining in the wild.",
      "Octopuses are intelligent invertebrates with eight arms. They can change color and texture to camouflage themselves. Octopuses have three hearts and blue blood.",
      "Hummingbirds are the smallest birds and can hover in mid-air. They beat their wings up to 80 times per second. Hummingbirds must eat every 10-15 minutes to maintain their energy.",
      "Sharks have existed for over 400 million years. They have cartilaginous skeletons instead of bones. Great white sharks can detect blood from miles away.",
      "Owls are nocturnal birds of prey with excellent night vision. Their feathers allow for silent flight while hunting. Owls can rotate their heads up to 270 degrees.",
      "Bees are essential pollinators that help plants reproduce. A single bee colony can contain up to 80,000 bees. Bees communicate through dance to share information about food sources.",
      "Butterflies undergo complete metamorphosis from caterpillar to adult. They taste with their feet and smell with their antennae. Some butterfly species migrate thousands of miles.",
      "Wolves are social predators that hunt in packs. They have complex communication systems including howling. Wolves are the ancestors of domestic dogs."
    ]

scores = retriever(question, documents)
best_idx = np.argmax(scores)
best_doc = documents[best_idx]
output = reader.generate(scores, question, [best_doc])
print(output)