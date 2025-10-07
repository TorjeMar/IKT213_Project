import requests

regnr = "EB54466"
r = requests.get('https://kjoretoyoppslag.atlas.vegvesen.no/ws/no/vegvesen/kjoretoy/kjoretoyoppslag/v2/oppslag/raw/'+ regnr)
#print all keys and values in the json response
data = r.json()
# print keys in the json response
print(f"Bredde : {data['kjoretoy']['godkjenning']['tekniskGodkjenning']['tekniskeData']['dimensjoner']['bredde'] / 10} cm")