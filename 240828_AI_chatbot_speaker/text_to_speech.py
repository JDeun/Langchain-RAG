from gtts import gTTS
from playsound import playsound

file_mane = 'sample.mp3'

# 영어 문장
# text = 'hello'
# tts_en = gTTS(text=text,lang='en')
# tts_en.save(file_mane)
# playsound(file_mane)

# 한글 문장
# text = '즐겁다'
# tts_ko = gTTS(text=text, lang='ko')
# tts_ko.save(file_mane)
# playsound(file_mane)

# 긴 문장
with open('sample.txt','r',encoding='utf8') as f:
    text = f.read()

tts_ko = gTTS(text=text, lang='ko')
tts_ko.save(file_mane)
playsound(file_mane)