def split(text):
    split_text = re.findall(r'\b\w+\b', text.lower())
    return split_text

def find_anser(context, anser):
    context_split = split(context)
    anser_split = split(anser)
    results = []
    for i in range(len(context_split)):
        if context_split[i] in anser_split:
            results.append(i)
    return results

context = "The Great Wall of China is one of the most iconic structures in the world, built over many centuries by various Chinese dynasties. The construction of this grand wall began in the 7th century BC and continued through several dynasties, notably the Ming Dynasty. Stretching an impressive length of approximately 13,000 miles; it runs across northern China and has become a significant symbol of Chinese history and culture."
context_split = context.split()
answer = "13,000 miles"
# , ; !
print(find_anser(context, answer))

from population import Individual
modified_indecies = [52]
words = ["Khoa"] * len(modified_indecies)
ind = Individual(
    context, 
    answer, 
    replacement_words=words, 
    modified_indices=modified_indecies
)
print(ind.get_perturbed_text())