def find_anser(context, answer):
    # finding position of the answer in context
    answer_split = [ans.lower() for ans in answer.split()]
    results = []
    context_split = context.split()
    for i in range(len(context_split)):
        for ans in answer_split:
            print(ans)
            if ans in context_split[i].lower():
                results.append(i)
    return results

context = "The Great Wall of China is one of the most iconic structures in the world, built over many centuries by various Chinese dynasties. The construction of this grand wall began in the 7th century BC and continued through several dynasties, notably the Ming Dynasty. Stretching an impressive length of approximately 13,000 miles, it runs across northern China and has become a significant symbol of Chinese history and culture."
context_split = context.split()
answer = "13,000 miles"

# print(find_anser(context, answer))

from population import Individual
modified_indecies = [38, 7, 45, 13, 16, 63, 32, 33, 36, 12, 20, 52, 27, 21, 6, 17, 47, 66, 58]
words = ["Khoa"] * len(modified_indecies)
ind = Individual(
    context, 
    answer, 
    replacement_words=words, 
    modified_indices=modified_indecies
)
print(ind.get_perturbed_text())