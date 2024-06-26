from datasets import load_dataset

# Load the WMT14 English-German dataset
dataset_en_de = load_dataset('wmt14', 'de-en')
# Load the WMT14 English-French dataset
dataset_en_fr = load_dataset('wmt14', 'fr-en')
# Load the WMT16 English-Romanian dataset
dataset_en_ro = load_dataset('wmt16', 'ro-en')

dataset_xsum = load_dataset('EdinburghNLP/xsum')

# You can access the data as follows:
print(dataset_en_de['train'][0])  # Show the first sample from the EN-DE train set
print(dataset_en_fr['train'][0])  # Show the first sample from the EN-FR train set
print(dataset_en_ro['train'][0])  # Show the first sample from the EN-RO train set
print(dataset_xsum['train'][0])  # Show the first sample from the XSum train set

print(dataset_en_de)
print(dataset_en_fr)
print(dataset_en_ro)
print(dataset_xsum)

def calculate_average_word_count(dataset, src_lang, tgt_lang, split='train'):
    source_word_counts = []
    target_word_counts = []

    # Iterate through each example in the specified split
    for example in dataset[split]:
        # Calculate word count for source and target sentences
        source_words = len(example['translation'][src_lang].split())
        target_words = len(example['translation'][tgt_lang].split())
        
        source_word_counts.append(source_words)
        target_word_counts.append(target_words)

    # Calculate averages
    average_source = sum(source_word_counts) / len(source_word_counts)
    average_target = sum(target_word_counts) / len(target_word_counts)

    return average_source, average_target

def calculate_average_length_xsum(dataset, src_lang, tgt_lang, split='train'):
    document_lengths = []
    summary_lengths = []

    # Iterate through each example in the dataset
    for example in dataset[split]:
        # Split the document and summary into words and count them
        document_words = len(example[src_lang].split())
        summary_words = len(example[tgt_lang].split())
        
        document_lengths.append(document_words)
        summary_lengths.append(summary_words)

    # Calculate the average lengths
    average_document_length = sum(document_lengths) / len(document_lengths)
    average_summary_length = sum(summary_lengths) / len(summary_lengths)

    return average_document_length, average_summary_length

# Calculate and print averages for each dataset
avg_en_de = calculate_average_word_count(dataset_en_de, "en", "de")
avg_en_fr = calculate_average_word_count(dataset_en_fr, "en", "fr")
avg_en_ro = calculate_average_word_count(dataset_en_ro, "en", "ro")
avg_xsum = calculate_average_length_xsum(dataset_xsum, "document", "summary")

print(f'Average Source and Target Word Counts for EN-DE: {avg_en_de}')
print(f'Average Source and Target Word Counts for EN-FR: {avg_en_fr}')
print(f'Average Source and Target Word Counts for EN-RO: {avg_en_ro}')
print(f'Average Source and Target Word Counts for XSum: {avg_xsum}')
