import whisper
import json
import os
model=whisper.load_model("large-v2")
audios=os.listdir("audios")   #for looping all the audios in audios file


# for all mp3
for audio in audios:
    if  audio.endswith(".mp3"):
        # print(audio)
        number=audio.split(".")[0]
        title=audio.split(".")[1]
        print(number,title)

        result=model.transcribe(audio=f"audios/{audio}",
        # result=model.transcribe(audio=f"audios/sample.mp3",
                                language="hi",
                                task="translate",
                                word_timestamps=False)
        
        chunks=[]
        for segment in result["segments"]:
            chunks.append({"number":number,"title":title, "start":segment["start"],"end":segment["end"],"text":segment["text"]})

        chunks_with_metadata={"chunks":chunks,"text":result["text"]}    

        # print(chunks)
        with open(f"jsons/{audio}.json","w") as f:
            json.dump(chunks_with_metadata,f)    
            