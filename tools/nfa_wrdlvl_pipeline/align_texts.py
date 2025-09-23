'''
This script performs a fuzzy alignment between two text files to create a new version of the original text
that only contains the segments present in the transcribed text.
'''

import re
import sys
from difflib import SequenceMatcher

def sanitize_text(text):
    '''Converts text to a list of sanitized words: lowercase, no punctuation.'''
    text = text.lower()
    text = re.sub(r'[".!,.:;?()-\[\]]', '', text)
    return text.split()

def levenshtein_distance(s1, s2):
    '''Calculates the Levenshtein distance between two strings.'''
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def find_fuzzy_matches_in_gap(original_words, transcript_words):
    '''
    Performs a fuzzy alignment on smaller segments (gaps) using a DP approach.
    Returns a set of indices from original_words that match.
    '''
    len_orig = len(original_words)
    len_trans = len(transcript_words)

    if len_orig == 0 or len_trans == 0:
        return set()

    def similarity(word1, word2):
        dist = levenshtein_distance(word1, word2)
        max_len = max(len(word1), len(word2))
        if max_len == 0: return 1.0
        return 1.0 - (dist / max_len)

    gap_penalty = -1
    dp = [[0] * (len_trans + 1) for _ in range(len_orig + 1)]
    traceback = [[0] * (len_trans + 1) for _ in range(len_orig + 1)]

    for i in range(1, len_orig + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
        traceback[i][0] = 2
    for j in range(1, len_trans + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        traceback[0][j] = 3

    for i in range(1, len_orig + 1):
        for j in range(1, len_trans + 1):
            orig_word = original_words[i-1]
            trans_word = transcript_words[j-1]
            
            sim = similarity(orig_word, trans_word)
            match_score = 2 * sim - 1

            diag_score = dp[i-1][j-1] + match_score
            up_score = dp[i-1][j] + gap_penalty
            left_score = dp[i][j-1] + gap_penalty

            scores = [diag_score, up_score, left_score]
            max_score = max(scores)
            dp[i][j] = max_score
            traceback[i][j] = scores.index(max_score) + 1

    matched_indices_in_gap = set()
    i, j = len_orig, len_trans
    while i > 0 and j > 0:
        move = traceback[i][j]
        if move == 1:
            if similarity(original_words[i-1], transcript_words[j-1]) > 0.6:
                matched_indices_in_gap.add(i-1)
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        elif move == 3:
            j -= 1
        else:
            break
    return matched_indices_in_gap

def main():
    original_file_path = 'input/Original_text.txt'
    transcript_file_path = 'input/Transcribed_text.txt'

    try:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
        with open(transcript_file_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure both files exist.", file=sys.stderr)
        sys.exit(1)

    original_tokens = re.split(r'(\s+)', original_text)
    sanitized_original_words = [re.sub(r'[\".!,.:;?()-\[\]]', '', w.lower()) for w in original_tokens if w.strip()]
    transcript_words = sanitize_text(transcript_text)

    # --- Pass 1: Exact matches using difflib ---
    matcher = SequenceMatcher(None, sanitized_original_words, transcript_words, autojunk=False)
    matching_blocks = list(matcher.get_matching_blocks())
    
    matched_original_indices = set()
    for i, j, n in matching_blocks:
        for k in range(n):
            matched_original_indices.add(i + k)

    # --- Pass 2: Fuzzy matches in gaps ---
    last_orig_end = 0
    last_trans_end = 0
    extended_blocks = matching_blocks + [(len(sanitized_original_words), len(transcript_words), 0)]

    for i, j, n in extended_blocks:
        orig_gap_start = last_orig_end
        orig_gap_end = i
        trans_gap_start = last_trans_end
        trans_gap_end = j

        if orig_gap_end > orig_gap_start and trans_gap_end > trans_gap_start:
            original_gap_words = sanitized_original_words[orig_gap_start:orig_gap_end]
            transcript_gap_words = transcript_words[trans_gap_start:trans_gap_end]
            
            matched_in_gap = find_fuzzy_matches_in_gap(original_gap_words, transcript_gap_words)
            
            for idx in matched_in_gap:
                matched_original_indices.add(orig_gap_start + idx)

        last_orig_end = i + n
        last_trans_end = j + n

    # --- Pass 3: Heuristic for word splits/merges in remaining gaps ---
    last_orig_end = 0
    last_trans_end = 0

    for i, j, n in extended_blocks:
        orig_gap_start = last_orig_end
        orig_gap_end = i
        trans_gap_start = last_trans_end
        trans_gap_end = j

        if orig_gap_end > orig_gap_start and trans_gap_end > trans_gap_start:
            is_fully_matched = all((orig_gap_start + k) in matched_original_indices for k in range(orig_gap_end - orig_gap_start))
            
            if not is_fully_matched:
                original_gap_words = sanitized_original_words[orig_gap_start:orig_gap_end]
                transcript_gap_words = transcript_words[trans_gap_start:trans_gap_end]
                
                orig_phrase = "".join(original_gap_words)
                trans_phrase = "".join(transcript_gap_words)
                
                if orig_phrase and trans_phrase:
                    dist = levenshtein_distance(orig_phrase, trans_phrase)
                    sim = 1.0 - (dist / max(len(orig_phrase), len(trans_phrase)))
                    
                    if sim >= 0.75:
                        for k in range(len(original_gap_words)):
                            matched_original_indices.add(orig_gap_start + k)

        last_orig_end = i + n
        last_trans_end = j + n

    # --- Reconstruct the final text ---
    result_text = []
    original_word_idx = 0
    for token in original_tokens:
        if token.strip():
            if original_word_idx in matched_original_indices:
                result_text.append(token)
            original_word_idx += 1
        else:
            if result_text and result_text[-1].strip():
                 if (original_word_idx - 1) in matched_original_indices:
                    result_text.append(token)
    
    final_output = "".join(result_text).strip()
    final_output = re.sub('\n{3,}', '\n\n', final_output)

    with open('Aligned_text.txt', 'w', encoding='utf-8') as f:
        f.write(final_output)

    print("Alignment complete. The result has been saved to Aligned_text.txt", file=sys.stderr)

if __name__ == "__main__":
    main()