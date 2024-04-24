import argparse
import re
import wikipedia

# run via $ python3 wiki-extract.py -p "San Francisco" > wiki.txt

def replace_periods(input_string):
    # Replace all occurrences of "word1.word2" with "word1 word2", due to weird wiki formatting
    return re.sub(r'(\w+)\.(\w+)', r'\1 \2', input_string)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--page', type=str, required=True, help='Wiki article title')
args = parser.parse_args()

wikipedia.set_lang('en')
page = wikipedia.page(args.page)
content = page.content

lines = content.split("\n")
s = ""
for line in lines:
	# Strip section headings
	if not line.startswith('='):
		line = replace_periods(line)
		s += line
print(s)
