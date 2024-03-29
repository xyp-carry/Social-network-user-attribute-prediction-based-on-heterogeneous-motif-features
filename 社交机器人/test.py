import re
ccc = 'RTe@999,assajo@asjias_ kwo@http://pbs.twimg.com/media/FL1HCjFVQBs-pc3'
ccc = 'RI @danspainhour: 0ur guy Kelin Parsons @Parsonskelin\n@Swaggy_T1232 https://t.co/JDwPYIFIyC'
words = re.findall(r'@\w+|\S+', ccc)
# words = re.findall(r'\S+|@\w+', ccc)
words = re.findall(r'\S+', ccc)
# words = re.findall(r'https?://[a-zA-Z0-9/.-]+', ccc)
if "RT" in words:
    print(1)
print(words)


print('appapp app'.count('app'))