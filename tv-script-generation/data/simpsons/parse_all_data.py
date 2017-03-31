import pandas as pd
df = pd.read_csv('simpsons_script_lines.csv', error_bad_lines=False)
df.drop(['id', 'episode_id', 'number', 'timestamp_in_ms', 'character_id', 'location_id', 'spoken_words', 'word_count'], axis=1, inplace=True)
df = df[df['speaking_line'] == True]
df.drop('speaking_line', axis=1, inplace=True)
df['raw_character_text'] = df['raw_character_text'].str.lower().replace(r'[^a-z]', '_')
df['line'] = df['raw_character_text'] + ': ' + df['normalized_text']
df.drop(['raw_character_text', 'normalized_text'], axis=1, inplace=True)
df.to_csv('all_lines.txt', columns=['line'], header=False, index=False)