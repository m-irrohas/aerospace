import csv
import codecs

with codecs.open("time_Temp.csv",encoding="utf-8") as tT_list:
  header = next(tT_list)
  t =[]
  T =[]
  for row in tT_list:
    if "time" in row:
      row = row.replace('\n','')
      row = row.replace('\r','')
      row = row.replace('time(sec)','')
      t.append(row)
print(t)


